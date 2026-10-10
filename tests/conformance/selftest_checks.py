"""Self-tests for the conformance scripts: each check must fail on a planted violation (a guard that cannot fail is
worthless) and pass on the repository."""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DocsCheck(unittest.TestCase):
    def run_in(self, files, license_file=False):
        mod = load("check_docs")
        with tempfile.TemporaryDirectory() as d:
            for name, text in files.items():
                p = os.path.join(d, name)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "w").write(text)
            if license_file:
                open(os.path.join(d, "LICENSE"), "w").write("x")
            mod.ROOT = d
            return mod.main()

    def test_clean_tree_passes(self):
        self.assertEqual(self.run_in({"README.md": "see [a](docs/a.md)", "docs/a.md": "hi"}), 0)

    def test_broken_link_fails(self):
        self.assertEqual(self.run_in({"README.md": "see [a](docs/missing.md)"}), 1)

    def test_link_in_code_fence_ignored(self):
        self.assertEqual(self.run_in({"README.md": "```\n[a](nope.md)\n```\n"}), 0)

    def test_open_source_claim_fails(self):
        self.assertEqual(self.run_in({"README.md": "ClusterLM is open-source."}), 1)

    def test_open_source_allowed_in_licensing_docs(self):
        self.assertEqual(self.run_in({"docs/licensing/x.md": "not open source yet"}), 0)

    def test_license_file_fails(self):
        self.assertEqual(self.run_in({"README.md": "x"}, license_file=True), 1)


class OfflineCheck(unittest.TestCase):
    def run_in(self, text):
        mod = load("check_offline")
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "orchestrator"))
            open(os.path.join(d, "orchestrator", "x.cpp"), "w").write(text)
            mod.ROOT = d
            return mod.main()

    def test_clean(self):
        self.assertEqual(self.run_in('int main(){return 0;}\n'), 0)

    def test_url_fails(self):
        self.assertEqual(self.run_in('const char* u = "https://example.com";\n'), 1)

    def test_curl_include_fails(self):
        self.assertEqual(self.run_in('#include <curl/curl.h>\n'), 1)


class Repository(unittest.TestCase):
    def test_scripts_pass_on_repo(self):
        for s in ("check_docs.py", "check_offline.py"):
            r = subprocess.run([sys.executable, "-I", os.path.join(HERE, s)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
