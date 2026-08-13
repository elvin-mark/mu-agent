import pytest

from mu_agent.patching import (
    apply_patch_string,
    fuzzy_replace_string,
    parse_unified_diff,
)


def test_parse_unified_diff():
    sample_patch = """--- a/src/app.py
+++ b/src/app.py
@@ -10,3 +10,3 @@
 def foo():
-    return 1
+    return 42
"""
    patches = parse_unified_diff(sample_patch)
    assert len(patches) == 1
    assert patches[0].new_path == "src/app.py"
    assert len(patches[0].hunks) == 1
    assert patches[0].hunks[0].old_start == 10


def test_vibe_style_patch_parsing(tmp_path):
    target_file = tmp_path / "fib.py"
    target_file.write_text(
        "a, b = 0, 1\nwhile a <= limit:\n    yield a\n    a, b = b, a + b\n"
    )

    vibe_patch = """*** Begin Patch
*** Update File: fib.py
@@
-a, b = 0, 1
+a, b = 0, 1
-while a <= limit:
+while a <= limit:
-    yield a
+    yield a
-    a, b = b, a + b
+    a, b = b, a + b
+
+def is_prime(n: int) -> bool:
+    return n > 1
*** End Patch"""

    ok, _msg = apply_patch_string(vibe_patch, root_dir=str(tmp_path))
    assert ok is True
    assert "is_prime" in target_file.read_text()


def test_vibe_patch_deep_in_file(tmp_path):
    target_file = tmp_path / "deep.py"
    # Create 30 lines of padding before target block
    padding = ["# line " + str(i) for i in range(30)]
    content = "\n".join(padding) + "\n\ndef target_func():\n    return 'old_value'\n"
    target_file.write_text(content)

    vibe_patch = """*** Begin Patch
*** Update File: deep.py
@@
-def target_func():
-    return 'old_value'
+def target_func():
+    return 'new_value'
*** End Patch"""

    ok, _msg = apply_patch_string(vibe_patch, root_dir=str(tmp_path))
    assert ok is True
    assert "new_value" in target_file.read_text()


def test_fuzzy_replace_string():
    content = "line 1\nline 2\nline 3\nline 4\n"
    # Exact match
    res, ok, mode = fuzzy_replace_string(
        content, "line 2\nline 3", "line 2_modified\nline 3"
    )
    assert ok is True
    assert mode == "Exact match"
    assert "line 2_modified" in res

    # Fuzzy whitespace match
    fuzzy_target = "line 2  \nline 3  "
    res_f, ok_f, mode_f = fuzzy_replace_string(
        content, fuzzy_target, "line 2_fuzzy\nline 3"
    )
    assert ok_f is True
    assert mode_f == "Fuzzy match"
    assert "line 2_fuzzy" in res_f


@pytest.mark.asyncio
async def test_apply_patch_file(tmp_path):
    target_file = tmp_path / "hello.py"
    target_file.write_text("def hello():\n    print('hello world')\n    return 0\n")

    patch_text = """--- a/hello.py
+++ b/hello.py
@@ -1,3 +1,3 @@
 def hello():
-    print('hello world')
+    print('hello universe')
     return 0
"""
    ok, msg = apply_patch_string(patch_text, root_dir=str(tmp_path))
    assert ok is True
    assert "1 hunk applied" in msg
    assert "hello universe" in target_file.read_text()
