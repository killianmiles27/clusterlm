#include "clusterlm/father/gguf_bpe_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <queue>

#include "unicode_tables.hpp"

namespace clusterlm::father {

namespace {

constexpr std::uint32_t kInvalidCp = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------------------------- GPT-2 byte map
// bytes_to_unicode() of GPT-2: printable Latin-1 bytes map to themselves, the other 68 bytes to U+0100.. in order.
struct ByteMap {
  std::array<std::string, 256> to_utf8;
  std::array<std::int16_t, 0x100 + 68> from_cp;  // code point -> byte, -1 when not part of the map

  ByteMap() {
    from_cp.fill(-1);
    auto printable = [](unsigned b) { return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255); };
    unsigned extra = 0;
    for (unsigned b = 0; b < 256; ++b) {
      const unsigned cp = printable(b) ? b : 256 + extra++;
      from_cp[cp] = static_cast<std::int16_t>(b);
      std::string s;
      if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
      } else {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      }
      to_utf8[b] = std::move(s);
    }
  }
  int byte_of(std::uint32_t cp) const { return cp < from_cp.size() ? from_cp[cp] : -1; }
};
const ByteMap& byte_map() {
  static const ByteMap m;
  return m;
}

// ---------------------------------------------------------------------------------------------- UTF-8
std::size_t utf8_len_from_lead(unsigned char c) {
  static constexpr std::size_t kLen[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  return kLen[c >> 4];
}

// Strict decoder: rejects overlong forms, surrogates and values above U+10FFFF. Returns false (consuming nothing)
// for anything else; the caller then treats the lead byte as one invalid unit.
bool decode_utf8(std::string_view s, std::size_t pos, std::uint32_t& cp, std::size_t& len) {
  const auto b0 = static_cast<unsigned char>(s[pos]);
  if (b0 < 0x80) { cp = b0; len = 1; return true; }
  auto cont = [&](std::size_t i) { return pos + i < s.size() && (static_cast<unsigned char>(s[pos + i]) & 0xC0) == 0x80; };
  auto bits = [&](std::size_t i) { return static_cast<std::uint32_t>(static_cast<unsigned char>(s[pos + i]) & 0x3F); };
  if (b0 >= 0xC2 && b0 <= 0xDF && cont(1)) { cp = ((b0 & 0x1Fu) << 6) | bits(1); len = 2; return true; }
  if (b0 >= 0xE0 && b0 <= 0xEF && cont(1) && cont(2)) {
    cp = ((b0 & 0x0Fu) << 12) | (bits(1) << 6) | bits(2);
    if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
    len = 3;
    return true;
  }
  if (b0 >= 0xF0 && b0 <= 0xF4 && cont(1) && cont(2) && cont(3)) {
    cp = ((b0 & 0x07u) << 18) | (bits(1) << 12) | (bits(2) << 6) | bits(3);
    if (cp < 0x10000 || cp > 0x10FFFF) return false;
    len = 4;
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------------------------- pre-tokenizer
enum class Kind : std::uint8_t { kLetter, kMark, kNumber, kSpace, kNewline, kOther };

struct Unit {
  std::uint32_t cp;   // kInvalidCp for a stray byte
  std::uint32_t off;  // byte offset
  std::uint8_t len;   // bytes
  Kind kind;
};

Kind kind_of(std::uint32_t cp) {
  if (cp == kInvalidCp) return Kind::kOther;
  if (cp == '\r' || cp == '\n') return Kind::kNewline;
  if (unicode::is_white_space(cp)) return Kind::kSpace;
  switch (unicode::class_of(cp)) {
    case unicode::Class::kLetter: return Kind::kLetter;
    case unicode::Class::kMark: return Kind::kMark;
    case unicode::Class::kNumber: return Kind::kNumber;
    case unicode::Class::kOther: break;
  }
  return Kind::kOther;
}

std::vector<Unit> to_units(std::string_view text) {
  std::vector<Unit> u;
  u.reserve(text.size());
  for (std::size_t pos = 0; pos < text.size();) {
    std::uint32_t cp = kInvalidCp;
    std::size_t len = 1;
    if (!decode_utf8(text, pos, cp, len)) {
      cp = kInvalidCp;
      len = 1;
    }
    u.push_back({cp, static_cast<std::uint32_t>(pos), static_cast<std::uint8_t>(len), kind_of(cp)});
    pos += len;
  }
  return u;
}

// Hand matcher for the qwen2 / qwen35 pattern. Every alternative is tried in order at a position (regex
// leftmost-first semantics, including the backtracking the pattern relies on); `marks_are_letters` is the one
// difference of qwen35 (\p{M} joins the letter class of alternatives 2 and 4).
class QwenSplitter {
 public:
  QwenSplitter(const std::vector<Unit>& u, bool marks_are_letters) : u_(u), n_(u.size()), marks_(marks_are_letters) {}

  // Length in units of the next match at p (always >= 1: every unit matches some alternative).
  std::size_t next(std::size_t p) const {
    std::size_t len = contraction(p);
    if (len) return len;
    if ((len = word(p))) return len;
    if (u_[p].kind == Kind::kNumber) return 1;
    if ((len = punct(p))) return len;
    if ((len = newline_run(p))) return len;
    if ((len = space_not_before_nonspace(p))) return len;
    if ((len = whitespace(p))) return len;
    return 1;  // unreachable for a well-formed class assignment; guarantees progress
  }

 private:
  bool is_space(std::size_t i) const { return u_[i].kind == Kind::kSpace || u_[i].kind == Kind::kNewline; }
  bool is_newline(std::size_t i) const { return u_[i].kind == Kind::kNewline; }
  bool letter_like(std::size_t i) const { return u_[i].kind == Kind::kLetter || (marks_ && u_[i].kind == Kind::kMark); }
  static std::uint32_t lower(std::uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

  // '[sS] | '[tT] | '[rR][eE] | '[vV][eE] | '[mM] | '[lL][lL] | '[dD]
  std::size_t contraction(std::size_t p) const {
    if (u_[p].cp != '\'' || p + 1 >= n_) return 0;
    const std::uint32_t c = lower(u_[p + 1].cp);
    if (c == 's' || c == 't' || c == 'm' || c == 'd') return 2;
    if (p + 2 >= n_) return 0;
    const std::uint32_t d = lower(u_[p + 2].cp);
    if ((c == 'r' && d == 'e') || (c == 'v' && d == 'e') || (c == 'l' && d == 'l')) return 3;
    return 0;
  }

  // [^\r\n\p{L}\p{N}]?[\p{L}(\p{M})]+
  std::size_t word(std::size_t p) const {
    const Kind k = u_[p].kind;
    const bool prefixable = k != Kind::kNewline && k != Kind::kLetter && k != Kind::kNumber;
    std::size_t q = p;
    if (prefixable && p + 1 < n_ && letter_like(p + 1)) {
      q = p + 1;
    } else if (!letter_like(p)) {
      return 0;
    }
    while (q < n_ && letter_like(q)) ++q;
    return q - p;
  }

  bool punct_class(std::size_t i) const {  // [^\s\p{L}(\p{M})\p{N}]
    const Kind k = u_[i].kind;
    return k == Kind::kOther || (!marks_ && k == Kind::kMark);
  }

  // ` ?[^\s\p{L}(\p{M})\p{N}]+[\r\n]*`
  std::size_t punct(std::size_t p) const {
    std::size_t q = p;
    if (u_[q].cp == ' ') ++q;
    if (q >= n_ || !punct_class(q)) return 0;
    while (q < n_ && punct_class(q)) ++q;
    while (q < n_ && is_newline(q)) ++q;
    return q - p;
  }

  std::size_t space_run_end(std::size_t p) const {
    std::size_t e = p;
    while (e < n_ && is_space(e)) ++e;
    return e;
  }

  // \s*[\r\n]+ : the greedy \s* backtracks to just before the last newline of the whitespace run.
  std::size_t newline_run(std::size_t p) const {
    const std::size_t e = space_run_end(p);
    for (std::size_t i = e; i > p; --i)
      if (is_newline(i - 1)) return i - p;
    return 0;
  }

  // \s+(?!\S) : the whole run at the end of the text, otherwise all but its last unit (which then starts the next word).
  std::size_t space_not_before_nonspace(std::size_t p) const {
    const std::size_t e = space_run_end(p);
    const std::size_t len = e - p;
    if (len == 0) return 0;
    if (e == n_) return len;
    return len >= 2 ? len - 1 : 0;
  }

  std::size_t whitespace(std::size_t p) const { return space_run_end(p) - p; }

  const std::vector<Unit>& u_;
  std::size_t n_;
  bool marks_;
};

bool is_eog_name(std::string_view t) {
  return t == "<|im_end|>" || t == "<|endoftext|>" || t == "<|eot_id|>" || t == "<|end_of_text|>" || t == "<|eom_id|>";
}

}  // namespace

// ---------------------------------------------------------------------------------------------- construction

objects::GgufLimits GgufBpeTokenizer::metadata_limits() {
  objects::GgufLimits l;
  l.allow_vocab_only_tail = true;
  l.retain_full_arrays = {"tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type"};
  return l;
}

Result<std::shared_ptr<GgufBpeTokenizer>> GgufBpeTokenizer::from_gguf_file(const std::filesystem::path& path,
                                                                           const GgufBpeOptions& options) {
  CLM_ASSIGN_OR_RETURN(auto file, objects::read_gguf_file(path, metadata_limits()));
  return from_gguf(file, options);
}

Result<std::shared_ptr<GgufBpeTokenizer>> GgufBpeTokenizer::from_gguf(const objects::GgufFile& file, const GgufBpeOptions& options) {
  std::shared_ptr<GgufBpeTokenizer> t(new GgufBpeTokenizer());
  CLM_RETURN_IF_ERROR(t->init(file, options));
  return t;
}

Status GgufBpeTokenizer::init(const objects::GgufFile& file, const GgufBpeOptions& options) {
  using objects::GgufValueType;
  auto fail = [](ErrorCode c, std::string m) { return make_error(c, "tokenizer: " + std::move(m)); };

  const auto* model = file.find("tokenizer.ggml.model");
  if (model == nullptr || !model->is_string()) return fail(ErrorCode::kInvalidArgument, "GGUF has no tokenizer.ggml.model");
  if (model->s != "gpt2")
    return fail(ErrorCode::kUnimplemented, "unsupported tokenizer model \"" + model->s + "\" (only byte-level BPE \"gpt2\" is implemented)");
  const auto* pre = file.find("tokenizer.ggml.pre");
  if (pre == nullptr || !pre->is_string())
    return fail(ErrorCode::kUnimplemented, "GGUF has no tokenizer.ggml.pre; the pre-tokenizer is never guessed");
  if (pre->s == "qwen2") pre_kind_ = Pre::kQwen2;
  else if (pre->s == "qwen35") pre_kind_ = Pre::kQwen35;
  else return fail(ErrorCode::kUnimplemented, "unsupported pre-tokenizer \"" + pre->s + "\" (implemented: qwen2, qwen35)");
  pre_ = pre->s;

  auto string_array = [&](const char* key, bool required, std::vector<std::string>& out) -> Status {
    const auto* v = file.find(key);
    if (v == nullptr) return required ? fail(ErrorCode::kInvalidArgument, std::string("GGUF has no ") + key) : Status::ok();
    if (!v->is_array() || v->elem_type != GgufValueType::kString)
      return fail(ErrorCode::kInvalidArgument, std::string(key) + " is not a string array");
    if (v->items.size() != v->count)
      return fail(ErrorCode::kInvalidArgument, std::string(key) + " was parsed without GgufBpeTokenizer::metadata_limits()");
    out.reserve(v->items.size());
    for (const auto& it : v->items) out.push_back(it.s);
    return Status::ok();
  };
  CLM_RETURN_IF_ERROR(string_array("tokenizer.ggml.tokens", true, tokens_));
  std::vector<std::string> merges;
  CLM_RETURN_IF_ERROR(string_array("tokenizer.ggml.merges", true, merges));
  if (tokens_.empty() || tokens_.size() > std::numeric_limits<std::int32_t>::max())
    return fail(ErrorCode::kInvalidArgument, "empty or oversized vocabulary");


  // Token types (llama.cpp's enum: 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte; others unused).
  attrs_.assign(tokens_.size(), Attr::kNormal);
  if (const auto* tt = file.find("tokenizer.ggml.token_type")) {
    if (!tt->is_array() || tt->items.size() != tt->count)
      return fail(ErrorCode::kInvalidArgument, "tokenizer.ggml.token_type is not a fully parsed array");
    if (tt->items.size() < tokens_.size())
      return fail(ErrorCode::kInvalidArgument, "tokenizer.ggml.token_type has fewer entries than tokens");
    for (std::size_t i = 0; i < tokens_.size(); ++i) {
      const auto v = tt->items[i].as_u64();
      switch (v.value_or(0)) {
        case 1: attrs_[i] = Attr::kNormal; break;
        case 2: attrs_[i] = Attr::kUnknown; break;
        case 3: attrs_[i] = Attr::kControl; break;
        case 4: attrs_[i] = Attr::kUserDefined; break;
        case 6: attrs_[i] = Attr::kByte; break;
        default: attrs_[i] = Attr::kUnused; break;
      }
    }
  }

  // Token text -> id (the last duplicate wins, as in llama.cpp); an empty token is a placeholder that is never matched.
  std::vector<bool> placeholder(tokens_.size(), false);
  token_to_id_.reserve(tokens_.size());
  for (std::size_t i = 0; i < tokens_.size(); ++i) {
    if (tokens_[i].empty()) {
      tokens_[i] = "[EMPTY_" + std::to_string(i) + "]";
      attrs_[i] = Attr::kUnused;
      placeholder[i] = true;
    }
    token_to_id_[tokens_[i]] = static_cast<std::int32_t>(i);
  }
  for (const auto& [text, id] : token_to_id_)
    if (is_eog_name(text) && !placeholder[static_cast<std::size_t>(id)]) attrs_[static_cast<std::size_t>(id)] = Attr::kControl;

  // The 256 GPT-2 byte tokens must exist: they guarantee that every byte string is representable and decodes back
  // exactly (llama.cpp silently drops bytes without a token; Father refuses such a vocabulary instead).
  const ByteMap& bm = byte_map();
  for (unsigned b = 0; b < 256; ++b)
    if (token_to_id_.find(bm.to_utf8[b]) == token_to_id_.end())
      return fail(ErrorCode::kInvalidArgument, "vocabulary lacks the byte-level base token for byte " + std::to_string(b));

  // Merge ranks: line index = priority (lower merges first); the first duplicate wins.
  ranks_.reserve(merges.size());
  for (std::size_t i = 0; i < merges.size(); ++i) {
    if (merges[i].find(' ', 1) == std::string::npos) continue;  // malformed line: never matchable
    ranks_.emplace(std::move(merges[i]), static_cast<std::int32_t>(i));
  }

  // Special-token cache: control, user-defined and unknown tokens, longest text first (ties by id).
  for (std::size_t i = 0; i < tokens_.size(); ++i)
    if (attrs_[i] == Attr::kControl || attrs_[i] == Attr::kUserDefined || attrs_[i] == Attr::kUnknown)
      special_order_.push_back(static_cast<std::int32_t>(i));
  std::sort(special_order_.begin(), special_order_.end(), [&](std::int32_t a, std::int32_t b) {
    const auto la = tokens_[static_cast<std::size_t>(a)].size(), lb = tokens_[static_cast<std::size_t>(b)].size();
    return la != lb ? la > lb : a < b;
  });

  auto id_key = [&](const char* key, std::optional<std::int32_t>& out) -> Status {
    const auto* v = file.find(key);
    if (v == nullptr) return Status::ok();
    const auto id = v->as_u64();
    if (!id || *id >= tokens_.size()) return fail(ErrorCode::kInvalidArgument, std::string(key) + " is not a valid token id");
    out = static_cast<std::int32_t>(*id);
    return Status::ok();
  };
  CLM_RETURN_IF_ERROR(id_key("tokenizer.ggml.bos_token_id", bos_));
  CLM_RETURN_IF_ERROR(id_key("tokenizer.ggml.eos_token_id", eos_));
  CLM_RETURN_IF_ERROR(id_key("tokenizer.ggml.eot_token_id", eot_));
  CLM_RETURN_IF_ERROR(id_key("tokenizer.ggml.padding_token_id", pad_));
  if (const auto* ab = file.find("tokenizer.ggml.add_bos_token")) add_bos_ = ab->type == GgufValueType::kBool && ab->u != 0;

  auto add_stop = [&](std::optional<std::int32_t> id) {
    if (id && std::find(stop_tokens_.begin(), stop_tokens_.end(), *id) == stop_tokens_.end()) stop_tokens_.push_back(*id);
  };
  add_stop(eos_);
  add_stop(eot_);
  add_stop(find_token("<|im_end|>"));
  add_stop(find_token("<|endoftext|>"));

  chat_options_ = options.chat;
  const auto* tmpl = file.find("tokenizer.chat_template");
  auto parsed = ChatMlTemplate::from_template_source(tmpl != nullptr && tmpl->is_string() ? std::string_view(tmpl->s) : std::string_view());
  if (parsed.is_ok()) {
    chat_ = std::move(parsed).value();
  } else if (options.require_chat_template) {
    return fail(parsed.status().code(), parsed.status().message());
  }
  return Status::ok();
}

// ---------------------------------------------------------------------------------------------- vocabulary access

std::string_view GgufBpeTokenizer::token_text(std::int32_t id) const {
  if (id < 0 || static_cast<std::size_t>(id) >= tokens_.size()) return {};
  return tokens_[static_cast<std::size_t>(id)];
}

std::optional<std::int32_t> GgufBpeTokenizer::find_token(std::string_view stored_text) const {
  auto it = token_to_id_.find(std::string(stored_text));
  if (it == token_to_id_.end()) return std::nullopt;
  return it->second;
}

bool GgufBpeTokenizer::is_special(std::int32_t id) const {
  if (id < 0 || static_cast<std::size_t>(id) >= tokens_.size()) return false;
  const Attr a = attrs_[static_cast<std::size_t>(id)];
  return a == Attr::kControl || a == Attr::kUserDefined;
}

// ---------------------------------------------------------------------------------------------- encode

std::vector<GgufBpeTokenizer::Fragment> GgufBpeTokenizer::partition_special(std::string_view text, bool parse_special) const {
  std::vector<Fragment> frags;
  if (!text.empty()) frags.push_back({false, 0, 0, text.size()});
  for (const std::int32_t id : special_order_) {
    const Attr a = attrs_[static_cast<std::size_t>(id)];
    if (!parse_special && (a == Attr::kControl || a == Attr::kUnknown)) continue;
    const std::string& st = tokens_[static_cast<std::size_t>(id)];
    if (text.find(st) == std::string_view::npos) continue;
    std::vector<Fragment> next;
    next.reserve(frags.size() + 2);
    for (const Fragment& f : frags) {
      if (f.is_token) {
        next.push_back(f);
        continue;
      }
      const std::string_view hay = text.substr(f.off, f.len);
      std::size_t p = 0;
      while (true) {
        const std::size_t m = hay.find(st, p);
        if (m == std::string_view::npos) {
          if (p < hay.size()) next.push_back({false, 0, f.off + p, hay.size() - p});
          break;
        }
        if (m > p) next.push_back({false, 0, f.off + p, m - p});
        next.push_back({true, id, 0, 0});
        p = m + st.size();
      }
    }
    frags = std::move(next);
  }
  return frags;
}

void GgufBpeTokenizer::bpe_word(std::string_view word, std::vector<std::int32_t>& out) const {
  struct Sym {
    int prev, next;
    std::size_t off, len;
  };
  struct Bigram {
    std::int32_t rank;
    int left, right;
    std::size_t llen, rlen;
  };
  struct Worse {  // priority_queue keeps the "largest": make that the lowest rank, then the leftmost symbol
    bool operator()(const Bigram& a, const Bigram& b) const { return a.rank > b.rank || (a.rank == b.rank && a.left > b.left); }
  };

  std::vector<Sym> syms;
  for (std::size_t off = 0; off < word.size();) {
    const std::size_t len = std::min(word.size() - off, utf8_len_from_lead(static_cast<unsigned char>(word[off])));
    const int idx = static_cast<int>(syms.size());
    syms.push_back({idx - 1, off + len == word.size() ? -1 : idx + 1, off, len});
    off += len;
  }
  std::priority_queue<Bigram, std::vector<Bigram>, Worse> queue;
  std::string key;
  auto add_bigram = [&](int l, int r) {
    if (l < 0 || r < 0) return;
    const Sym& a = syms[static_cast<std::size_t>(l)];
    const Sym& b = syms[static_cast<std::size_t>(r)];
    key.assign(word.data() + a.off, a.len);
    key.push_back(' ');
    key.append(word.data() + b.off, b.len);
    auto it = ranks_.find(key);
    if (it != ranks_.end()) queue.push({it->second, l, r, a.len, b.len});
  };
  for (int i = 1; i < static_cast<int>(syms.size()); ++i) add_bigram(i - 1, i);

  while (!queue.empty()) {
    const Bigram bg = queue.top();
    queue.pop();
    Sym& l = syms[static_cast<std::size_t>(bg.left)];
    Sym& r = syms[static_cast<std::size_t>(bg.right)];
    if (l.len == 0 || r.len == 0 || l.len != bg.llen || r.len != bg.rlen) continue;  // outdated
    l.len += r.len;
    r.len = 0;
    l.next = r.next;
    if (r.next >= 0) syms[static_cast<std::size_t>(r.next)].prev = bg.left;
    add_bigram(l.prev, bg.left);
    add_bigram(bg.left, l.next);
  }

  std::string piece;
  for (const Sym& s : syms) {
    if (s.len == 0) continue;
    piece.assign(word.data() + s.off, s.len);
    auto it = token_to_id_.find(piece);
    if (it != token_to_id_.end()) {
      out.push_back(it->second);
      continue;
    }
    // A merge result without a vocabulary entry (malformed vocab): fall back to its characters, which all exist.
    for (std::size_t o = 0; o < piece.size();) {
      const std::size_t len = std::min(piece.size() - o, utf8_len_from_lead(static_cast<unsigned char>(piece[o])));
      auto ch = token_to_id_.find(piece.substr(o, len));
      if (ch != token_to_id_.end()) out.push_back(ch->second);
      o += len;
    }
  }
}

void GgufBpeTokenizer::tokenize_plain(std::string_view text, std::vector<std::int32_t>& out) const {
  if (text.empty()) return;
  const std::vector<Unit> units = to_units(text);
  const QwenSplitter splitter(units, pre_kind_ == Pre::kQwen35);
  const ByteMap& bm = byte_map();
  std::string word;
  for (std::size_t p = 0; p < units.size();) {
    const std::size_t n = splitter.next(p);
    const std::size_t byte_begin = units[p].off;
    const std::size_t byte_end = units[p + n - 1].off + units[p + n - 1].len;
    word.clear();
    for (std::size_t i = byte_begin; i < byte_end; ++i) word += bm.to_utf8[static_cast<unsigned char>(text[i])];
    bpe_word(word, out);
    p += n;
  }
}

std::vector<std::int32_t> GgufBpeTokenizer::encode_text(std::string_view text, bool parse_special) const {
  std::vector<std::int32_t> out;
  out.reserve(text.size() / 3 + 1);
  for (const Fragment& f : partition_special(text, parse_special)) {
    if (f.is_token) out.push_back(f.token);
    else tokenize_plain(text.substr(f.off, f.len), out);
  }
  return out;
}

std::vector<std::int32_t> GgufBpeTokenizer::encode_chat(std::span<const ChatMessage> messages) const {
  return encode_chat(messages, chat_options_);
}

std::vector<std::int32_t> GgufBpeTokenizer::encode_chat(std::span<const ChatMessage> messages, const ChatTemplateOptions& options) const {
  std::vector<std::int32_t> out;
  if (!chat_) return out;
  if (add_bos_ && bos_) out.push_back(*bos_);
  std::string pending;  // adjacent plain pieces are one text, as in the template's output string
  auto flush = [&] {
    if (pending.empty()) return;
    const auto ids = encode_text(pending, false);
    out.insert(out.end(), ids.begin(), ids.end());
    pending.clear();
  };
  for (const ChatPiece& piece : chat_->render(messages, options)) {
    if (piece.marker) {
      const auto id = find_token(piece.text);
      if (id && is_special(*id)) {
        flush();
        out.push_back(*id);
        continue;
      }
    }
    pending += piece.text;
  }
  flush();
  return out;
}

// ---------------------------------------------------------------------------------------------- decode

void GgufBpeTokenizer::decode_into(std::span<const std::int32_t> ids, std::string& out) const {
  const ByteMap& bm = byte_map();
  for (const std::int32_t id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= tokens_.size()) continue;
    const std::string& text = tokens_[static_cast<std::size_t>(id)];
    const Attr attr = attrs_[static_cast<std::size_t>(id)];
    switch (attr) {
      case Attr::kControl:
      case Attr::kUnknown:
      case Attr::kUnused:
        break;  // renders as nothing
      case Attr::kUserDefined:
        out += text;
        break;
      case Attr::kByte:
      case Attr::kNormal: {
        if (attr == Attr::kByte && text.size() == 6 && text.compare(0, 3, "<0x") == 0 && text[5] == '>') {
          const auto hex = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'A' && c <= 'F' ? c - 'A' + 10 : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; };
          const int hi = hex(text[3]), lo = hex(text[4]);
          if (hi >= 0 && lo >= 0) {
            out.push_back(static_cast<char>(hi * 16 + lo));
            break;
          }
        }
        for (std::size_t o = 0; o < text.size();) {
          std::uint32_t cp = kInvalidCp;
          std::size_t len = 1;
          if (!decode_utf8(text, o, cp, len)) {
            cp = kInvalidCp;
            len = 1;
          }
          const int b = cp == kInvalidCp ? -1 : bm.byte_of(cp);
          if (b >= 0) out.push_back(static_cast<char>(b));
          else out.append(text, o, len);  // not a byte-level character: keep the stored bytes
          o += len;
        }
        break;
      }
    }
  }
}

std::string GgufBpeTokenizer::decode(std::span<const std::int32_t> tokens) const {
  std::string out;
  out.reserve(tokens.size() * 4);
  decode_into(tokens, out);
  return out;
}

}  // namespace clusterlm::father
