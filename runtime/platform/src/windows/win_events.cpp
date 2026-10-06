#ifdef _WIN32

#include "win_events.hpp"

#include <wtsapi32.h>

#include <cstring>

namespace clusterlm::platform::detail {

// {5d3e9a59-e9d5-4b00-a6bd-ff34ff516548}
const GUID kGuidAcDcPowerSource = {0x5d3e9a59, 0xe9d5, 0x4b00, {0xa6, 0xbd, 0xff, 0x34, 0xff, 0x51, 0x65, 0x48}};
// {E00958C0-C213-4ACE-AC77-FECCED2EEEA5}
const GUID kGuidPowerSavingStatus = {0xe00958c0, 0xc213, 0x4ace, {0xac, 0x77, 0xfe, 0xcc, 0xed, 0x2e, 0xee, 0xa5}};

std::optional<PowerEvent> map_power_broadcast(DWORD event_type, const void* data) {
  PowerEvent ev;
  switch (event_type) {
    case PBT_APMSUSPEND:
      ev.kind = PowerEventKind::kSuspend;
      return ev;
    case PBT_APMRESUMEAUTOMATIC:
    case PBT_APMRESUMESUSPEND:
      ev.kind = PowerEventKind::kResume;
      return ev;
    case PBT_POWERSETTINGCHANGE: {
      if (data == nullptr) return std::nullopt;
      const auto* s = static_cast<const POWERBROADCAST_SETTING*>(data);
      if (s->DataLength < sizeof(DWORD)) return std::nullopt;
      DWORD value = 0;
      std::memcpy(&value, s->Data, sizeof value);
      if (IsEqualGUID(s->PowerSetting, kGuidAcDcPowerSource)) {
        ev.kind = PowerEventKind::kPowerSourceChanged;
        ev.on_ac_power = (value == 0);  // 0 = AC, 1 = DC (battery), 2 = short-term DC (UPS)
        return ev;
      }
      if (IsEqualGUID(s->PowerSetting, kGuidPowerSavingStatus)) {
        ev.kind = PowerEventKind::kPowerSavingChanged;
        ev.battery_saver = (value != 0);
        return ev;
      }
      return std::nullopt;
    }
    default: return std::nullopt;
  }
}

std::optional<SessionEvent> map_session_change(DWORD event_type, DWORD session_id) {
  SessionEvent ev;
  ev.session_id = session_id;
  switch (event_type) {
    case WTS_SESSION_LOGON: ev.kind = SessionEventKind::kLogon; break;
    case WTS_SESSION_LOGOFF: ev.kind = SessionEventKind::kLogoff; break;
    case WTS_SESSION_LOCK: ev.kind = SessionEventKind::kLock; break;
    case WTS_SESSION_UNLOCK: ev.kind = SessionEventKind::kUnlock; break;
    case WTS_CONSOLE_CONNECT:
    case WTS_REMOTE_CONNECT: ev.kind = SessionEventKind::kConnect; break;
    case WTS_CONSOLE_DISCONNECT:
    case WTS_REMOTE_DISCONNECT: ev.kind = SessionEventKind::kDisconnect; break;
    default: return std::nullopt;
  }
  return ev;
}

}  // namespace clusterlm::platform::detail

#endif  // _WIN32
