"""Tests for the Hugging Face cache lookup (#506): setup takes a model out of the local huggingface_hub cache
instead of downloading it again - the pinned revision only, the blob the snapshot symlink points at, a hardlink so
the model is not stored twice, and no request at all (the tests fail if urlopen is touched).

    python -m unittest tools.test_setup_hf_cache
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

REV = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"          # HF_REVISIONS[the original GGUF repo]
OTHER = "f" * 40
FILE = "IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"   # a repository subfolder, as the coder's is
URL = f"https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/{REV}/{FILE}"


def no_network(*a, **kw):
    raise AssertionError("touched the network")


class CacheCase(unittest.TestCase):
    """A hub cache with one whole file (a symlink into blobs/, the way huggingface_hub writes it) plus a second
    revision's snapshot of the same repository."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)
        self.blob = self.cache / "models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF" / "blobs" / "e11083b"
        self.blob.parent.mkdir(parents=True)
        self.blob.write_bytes(b"GGUF-bytes" * 100)
        self.snap = self.cache / "models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF" / "snapshots"
        link(self.snap / REV / FILE, self.blob)
        link(self.snap / OTHER / FILE, self.blob)             # another revision: the same name, not our bytes
        (self.snap / REV / "model.safetensors.index.json").write_bytes(b"{}")
        self.out = self.cache / "out"
        self.out.mkdir()
        self.env = mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.cache)}, clear=False)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def run_silently(self, fn, *a, **kw):
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            fn(*a, **kw)
        return out.getvalue()


def link(path: Path, target: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.symlink_to(target)
    except OSError:                       # Windows without developer mode: the hub then copies the blob
        path.write_bytes(target.read_bytes())


class CacheFolder(CacheCase):
    def test_the_folder_is_read_as_huggingface_hub_reads_it(self):
        self.assertEqual(setup.hf_hub_cache(), self.cache)                       # $HF_HUB_CACHE wins
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": "", "HUGGINGFACE_HUB_CACHE": "/old"}):
            self.assertEqual(setup.hf_hub_cache(), Path("/old"))
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": "", "HUGGINGFACE_HUB_CACHE": "", "HF_HOME": "/h"}):
            self.assertEqual(setup.hf_hub_cache(), Path("/h/hub"))
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": "", "HUGGINGFACE_HUB_CACHE": "", "HF_HOME": "",
                                          "XDG_CACHE_HOME": "/x"}):
            self.assertEqual(setup.hf_hub_cache(), Path("/x/huggingface/hub"))
        blank = {"HF_HUB_CACHE": "", "HUGGINGFACE_HUB_CACHE": "", "HF_HOME": "", "XDG_CACHE_HOME": ""}
        with mock.patch.dict(os.environ, blank):
            self.assertEqual(setup.hf_hub_cache(), Path.home() / ".cache" / "huggingface" / "hub")

    def test_blanks_and_spaces_are_no_folder(self):
        for value in ("", "   "):
            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": value}):
                self.assertNotEqual(setup.hf_hub_cache(), Path(value))            # not the empty path


class Lookup(CacheCase):
    def test_the_pinned_revision_is_found_through_the_symlink(self):
        found = setup.hf_cached(URL)
        self.assertEqual(found, self.snap / REV / FILE)
        self.assertEqual(found.read_bytes(), self.blob.read_bytes())

    def test_another_revision_is_not_used(self):
        # the pinned revision is NOT in the cache, another one is (refs/main points there): it does not answer
        (self.snap / REV / FILE).unlink()
        (self.snap / REV / FILE).parent.rmdir()
        self.assertTrue((self.snap / OTHER / FILE).exists())
        self.assertIsNone(setup.hf_cached(URL))
        self.assertEqual(setup.hf_cached(URL.replace(REV, OTHER)), self.snap / OTHER / FILE)

    def test_a_mirror_url_resolves_to_the_same_cache(self):
        # huggingface_hub keys the cache by repo+revision, not by host, so a file fetched through
        # HF_ENDPOINT=hf-mirror.com sits in this same snapshot and passes the same revision and sha256 checks
        mirror = URL.replace("https://huggingface.co", "https://hf-mirror.com")
        self.assertEqual(setup.hf_cached(mirror), self.snap / REV / FILE)

    def test_urls_without_a_pinned_revision_are_not_guessed(self):
        main = URL.replace("/resolve/%s/" % REV, "/resolve/main/")
        self.assertIsNone(setup.hf_cached(main))                     # main is not this checkout's revision

    def test_other_urls_are_left_to_the_downloader(self):
        for url in ("https://github.com/ggml-org/llama.cpp/archive/x.zip", "file:///tmp/a.gguf",
                    "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/%s/x.gguf" % "abc",
                    URL + ".missing",
                    "https://huggingface.co/other-org/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/%s/%s" % (REV, FILE)):
            with self.subTest(url=url):
                self.assertIsNone(setup.hf_cached(url))              # not a model file at a pinned HF revision

    def test_a_file_the_hub_has_not_finished_is_not_there(self):
        # huggingface_hub writes blobs/<etag>.incomplete while downloading and only links it into snapshots/ when
        # the whole file is there - so a snapshot entry is a whole file, and nothing else has to be checked
        whole = self.snap / REV / FILE
        whole.unlink()
        self.assertIsNone(setup.hf_cached(URL))

    def test_a_file_in_another_subfolder_is_found_by_its_name(self):
        moved = self.snap / REV / "IQ3_XXS-old" / Path(FILE).name
        link(moved, self.blob)
        (self.snap / REV / FILE).unlink()
        self.assertEqual(setup.hf_cached(URL), moved)

    def test_two_candidates_are_not_a_guess(self):
        link(self.snap / REV / "a" / Path(FILE).name, self.blob)
        link(self.snap / REV / "b" / Path(FILE).name, self.blob)
        (self.snap / REV / FILE).unlink()
        self.assertIsNone(setup.hf_cached(URL))                      # the network answers better than a coin flip


class Take(CacheCase):
    def dst(self):
        return self.out / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"

    def test_it_is_a_hardlink_and_marked(self):
        self.assertTrue(setup.take_from_hf_cache(URL, self.dst(), "the model"))
        dst = self.dst()
        self.assertEqual(dst.read_bytes(), self.blob.read_bytes())
        self.assertEqual(dst.stat().st_ino, self.blob.stat().st_ino, "stored twice: 58 GB of free disk")
        self.assertTrue(setup.done(dst))
        self.assertFalse(list(self.out.glob("*.from-hf")))            # the intermediate name is gone

    def test_a_copied_blob_when_the_link_is_refused(self):
        with mock.patch.object(setup.os, "link", side_effect=OSError(18, "Invalid cross-device link")):
            self.assertTrue(setup.take_from_hf_cache(URL, self.dst()))
        self.assertEqual(self.dst().read_bytes(), self.blob.read_bytes())
        self.assertTrue(setup.done(self.dst()))

    def test_an_unfinished_file_next_to_it_does_not_stop_the_take(self):
        self.dst().write_bytes(b"half a model")                        # a leftover without its mark
        self.run_silently(setup.take_from_hf_cache, URL, self.dst())
        self.assertEqual(self.dst().read_bytes(), self.blob.read_bytes())

    def test_nothing_cached_says_so_and_leaves_the_file_alone(self):
        (self.snap / REV / FILE).unlink()                            # nothing for this revision in the cache
        self.assertFalse(setup.take_from_hf_cache(URL, self.dst()))
        self.assertFalse(self.dst().exists())
        self.assertFalse(setup.done(self.dst()))


class Download(CacheCase):
    def test_a_hugging_face_url_is_never_downloaded(self):
        dst = self.out / "shard.gguf"
        with mock.patch.object(setup.urllib.request, "urlopen", no_network):     # the download would need this
            self.run_silently(setup.download, URL, dst, "the model")
        self.assertEqual(dst.read_bytes(), self.blob.read_bytes())
        self.assertTrue(setup.done(dst))

    def test_a_second_run_skips_it_before_the_cache_is_even_read(self):
        dst = self.out / "shard.gguf"
        self.run_silently(setup.download, URL, dst)
        with mock.patch.object(setup, "hf_cached", side_effect=AssertionError("read the cache again")):
            text = self.run_silently(setup.download, URL, dst)
        self.assertIn("already downloaded", text)

    def test_a_file_url_still_copies(self):
        src = self.cache / "hand.gguf"
        src.write_bytes(b"downloaded by hand")
        dst = self.out / "hand.gguf"
        self.run_silently(setup.download, str(src), dst)
        self.assertEqual(dst.read_bytes(), b"downloaded by hand")
        self.assertTrue(setup.done(dst))


class Gate(unittest.TestCase):
    """#506: the free-disk gate asks the hub cache before it asks the drive - a model that is only hardlinked from the
    cache needs no room for itself, or setup stops a machine that already has it."""

    SIZES = (29_608_446_496, 28_800_138_432)            # the two real IQ1_M shards: 58 408 884 928 B

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.cache = self.out / "hub"
        self.fam = setup.FAMILIES["coder"]
        repo = self.fam["hf"].split("/resolve/")[0].split("/", 3)[3]
        rev = setup.HF_RESOLVE.match(self.fam["hf"].format(q="IQ1_M"))["rev"]
        self.shards = [self.out / self.fam["file"].format(q="IQ1_M", i=i) for i in (1, 2)]
        self.snap = self.cache / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
        for i, (s, size) in enumerate(zip(self.shards, self.SIZES), 1):
            blob = self.cache / "blobs" / f"blob{i}"
            blob.parent.mkdir(parents=True, exist_ok=True)
            with open(blob, "wb") as f:
                f.truncate(size)                         # sparse: the size is what counts, not 58 GB of RAM
            link(self.snap / self.fam["file"].format(q="IQ1_M", i=i), blob)
        self.env = mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.cache)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_the_pinned_cached_shards_count_as_room_that_is_already_there(self):
        with mock.patch.object(setup, "same_filesystem", return_value=False):
            self.assertEqual(setup.cached_bytes(self.shards, self.fam, "IQ1_M", self.out), 0)
        with mock.patch.object(setup, "same_filesystem", return_value=True):
            cached = setup.cached_bytes(self.shards, self.fam, "IQ1_M", self.out)
        self.assertAlmostEqual(cached, 58.4, delta=0.2)          # 58 408 884 928 B, the two real sizes
        # the gate stops asking for room for the model itself
        self.assertLess(setup.disk_need("IQ1_M", "coder", False, "none", True, cached), 35)
        self.assertGreater(setup.disk_need("IQ1_M", "coder", False, "none", True, 0), 90)

    def test_a_cache_on_another_drive_counts_the_bytes_again(self):
        with mock.patch.object(setup, "same_filesystem", return_value=False):
            self.assertEqual(setup.cached_bytes(self.shards, self.fam, "IQ1_M", self.out), 0)

    def test_a_finished_shard_is_counted_once_not_twice(self):
        dst = self.shards[0]
        with open(dst, "wb") as f:
            f.truncate(100)
        setup.mark(dst)
        with mock.patch.object(setup, "same_filesystem", return_value=True):
            self.assertAlmostEqual(setup.cached_bytes([dst], self.fam, "IQ1_M", self.out), 100 / 1e9)

    def test_a_part_file_counts(self):
        (self.out / (self.shards[0].name + ".part")).write_bytes(b"\0" * 1_000_000)
        with mock.patch.object(setup, "same_filesystem", return_value=False):
            self.assertAlmostEqual(setup.cached_bytes([self.shards[0]], self.fam, "IQ1_M", self.out), 0.001, places=3)

    def test_gguf_dir_is_never_measured_against_the_cache(self):
        with mock.patch.object(setup, "same_filesystem", return_value=True):
            self.assertEqual(setup.cached_bytes(self.shards, self.fam, "IQ1_M", self.out, gguf_dir=True), 0)

    def test_the_numbers_of_the_low_ram_mode_add_up(self):
        """Q2_0 on a 32 GB PC with no AVX-512: the experts become their own 34 GB file, so 109 GB in all (#506)."""
        need = setup.disk_need("Q2_0", "qwen", False, "none", True, 0)
        self.assertAlmostEqual(need, 66.4 + 8 + 34 + 1)
        self.assertEqual(f"{need:.0f}", "109")                     # the message the user was shown
        with_avx512 = setup.disk_need("Q2_0", "qwen", True, "none", False, 0)
        self.assertAlmostEqual(with_avx512, 66.4 + 8 + 40)         # the rewrite replaces the experts file
        self.assertEqual(setup.disk_need("Q2_0", "qwen", False, "none", True, 0, True), 8 + 34 + 1)

    def test_what_is_there_is_not_paid_for_twice(self):
        full = setup.MODELS["IQ3_XXS"]["download_gb"]
        self.assertAlmostEqual(setup.disk_need("IQ3_XXS", "qwen", False, "none", True, full), 8 + 42.9 + 1, places=1)
        self.assertAlmostEqual(setup.disk_need("IQ3_XXS", "qwen", False, "none", True, full + 50), 8 + 42.9 + 1)


class WhatIsRecommended(unittest.TestCase):
    """#506: a model that is already in the hub cache is what setup recommends when nothing else can be used - the
    owner keeps every other choice, it is only the default that moves."""

    def test_without_a_cache_the_ram_rule_still_picks_the_size(self):
        for plain in setup.family_sizes("qwen"):
            self.assertEqual(setup.cached_pick(setup.family_sizes("qwen"), {}, plain, 31.0, 16.0), (plain, False))

    def test_a_cached_size_of_the_same_family_wins_over_the_ram_rule(self):
        names = setup.family_sizes("qwen")
        self.assertEqual(setup.cached_pick(names, {"IQ2_XS": True}, "IQ3_XXS", 31.0, 16.0),
                         ("IQ2_XS", True))
        # the RAM rule's own pick is cached: nothing moves
        self.assertEqual(setup.cached_pick(names, {"qwen": {"IQ3_XXS": True}}, "IQ3_XXS", 64.0, 24.0),
                         ("IQ3_XXS", False))

    def test_a_family_with_a_cached_model_wins_over_the_first_one(self):
        fams = list(setup.FAMILIES)
        # a family: it counts when one of its cached sizes fits here
        self.assertEqual(setup.cached_pick(fams, {"coder": {"IQ1_M": True}}, fams[0], 31.0, 16.0,
                                           runs=lambda f: any(setup.runs_here(m, 31.0, 16.0) for m in {"IQ1_M"})),
                         ("coder", True))
        self.assertEqual(setup.cached_pick(fams, {"coder": {"IQ1_M": True}}, fams[0], 31.0, 16.0,
                                           runs=lambda f: False), (fams[0], False))
        self.assertEqual(setup.cached_pick(fams, {}, fams[0], 31.0, 16.0), (fams[0], False))

    def test_a_model_this_pc_cannot_run_is_never_recommended(self):
        # IQ3_XXS needs 60 GB RAM; a 16 GB PC that cannot use the low-RAM mode must not be pointed at it
        with mock.patch.object(setup, "low_ram_fits", return_value=False):
            names = setup.family_sizes("qwen")
            self.assertEqual(setup.cached_pick(names, {"qwen": {"IQ3_XXS": True}}, "Q2_0", 20.0, 16.0),
                             ("Q2_0", False))
            self.assertEqual(setup.cached_pick(names, {"Q2_0": True}, "Q2_0", 20.0, 16.0), ("Q2_0", False))
        # on this PC (31 GB RAM, a 16 GB card) the low-RAM mode fits IQ2_XS but not IQ3_XXS (60 GB of RAM), so
        # only the one that fits may be recommended - measured with setup's own rule
        self.assertFalse(setup.runs_here("IQ3_XXS", 31.0, 16.0))
        self.assertEqual(setup.cached_pick(setup.family_sizes("qwen"), {"IQ2_XS": True, "IQ3_XXS": True},
                                           "Q2_0", 31.0, 16.0), ("IQ2_XS", True))

    def test_the_cache_counts_only_at_the_pinned_revision(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            cache = out / "hub"
            fam = setup.FAMILIES["coder"]
            repo = fam["hf"].split("/resolve/")[0].split("/", 3)[3]
            folder = repo.replace("/", "--")
            rev = setup.HF_REVISIONS[repo]
            (cache / "blobs").mkdir(parents=True)
            for b in ("b1", "b2", "b3"):
                (cache / "blobs" / b).write_bytes(b"x")   # what the snapshot symlinks point at

            def snap(rev, i):
                return cache / ("models--" + folder) / "snapshots" / rev / "IQ1_M" / fam["file"].format(q="IQ1_M", i=i)

            link(snap(rev, 1), cache / "blobs" / "b1")
            link(snap("0" * 40, 1), cache / "blobs" / "b2")           # another revision, must not count
            env = mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(cache)})
            env.start()
            try:
                self.assertFalse(setup.in_hf_cache("coder", "IQ1_M"))       # one shard short
                self.assertFalse(setup.in_hf_cache("qwen", "IQ3_XXS"))      # not in this cache at all
                self.assertEqual(setup.cached_sizes("coder", out), {})      # and so not recommended
                link(snap(rev, 2), cache / "blobs" / "b3")
                self.assertTrue(setup.in_hf_cache("coder", "IQ1_M"))
                self.assertEqual(setup.cached_sizes("coder", out), {"IQ1_M": True})     # one filesystem
                self.assertEqual(setup.cached_sizes("coder", Path("/mnt/other-drive")), {"IQ1_M": False})
            finally:
                env.stop()


class Drives(unittest.TestCase):
    def test_same_filesystem(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            self.assertTrue(setup.same_filesystem(p, p / "x"))
            self.assertTrue(setup.same_filesystem(p / "a/b/c", p / "a"))     # a folder that is not there yet
            self.assertFalse(setup.same_filesystem(p, Path("/nonexistent/drive")))
        # two real filesystems, if the machine has more than one
        roots = {os.stat(m).st_dev for m in ("/mnt", "/media") if os.path.isdir(m)}
        if len(roots) > 1:
            self.assertFalse(setup.same_filesystem(Path("/mnt"), Path("/media")))

    def test_bigger_places_names_mounts_with_room_biggest_first(self):
        with tempfile.TemporaryDirectory() as d:
            big, small = Path(d) / "big", Path(d) / "small"
            big.mkdir()
            small.mkdir()
            usage = {str(big): 500 * 1e9, str(small): 120 * 1e9}
            with mock.patch.object(setup.shutil, "disk_usage",
                                   side_effect=lambda p: mock.Mock(free=usage[str(p)])), \
                    mock.patch.object(setup.os.path, "ismount", side_effect=lambda p: True), \
                    mock.patch.object(setup.Path, "iterdir", autospec=True,
                                      side_effect=lambda self: iter([big, small]) if str(self) == "/mnt" else iter([])):
                self.assertEqual(setup.bigger_places(100, Path("/nowhere")), [str(big), str(small)])
                self.assertEqual(setup.bigger_places(200, Path("/nowhere")), [str(big)])
                self.assertEqual(setup.bigger_places(600, Path("/nowhere")), [])
                self.assertEqual(setup.bigger_places(100, big), [str(small)])      # the folder itself is left out

    def test_bigger_places_ignores_what_is_not_a_mount(self):
        with tempfile.TemporaryDirectory() as d:
            plain = Path(d) / "plain"
            plain.mkdir()
            with mock.patch.object(setup.shutil, "disk_usage", side_effect=lambda p: mock.Mock(free=500 * 1e9)), \
                    mock.patch.object(setup.os.path, "ismount", return_value=False), \
                    mock.patch.object(setup.Path, "iterdir", autospec=True,
                                      side_effect=lambda self: iter([plain]) if str(self) == "/mnt" else iter([])):
                self.assertEqual(setup.bigger_places(100, Path("/nowhere")), [])


class RealRepositories(unittest.TestCase):
    """The cache layout of the repositories setup itself downloads: every model's real download URL must be one the
    lookup can answer, down to the file's path inside the repository (the coder's are in a subfolder)."""

    @staticmethod
    def url(fam, model, i=1):
        return fam["hf"].format(q=model) + fam["file"].format(q=model, i=i)      # as step 5 builds it

    def test_every_model_url_is_a_pinned_hugging_face_one(self):
        seen = 0
        for name, fam in setup.FAMILIES.items():
            for model, spec in setup.MODELS.items():
                if name not in spec.get("families", ("qwen",)):
                    continue
                seen += 1
                with self.subTest(family=name, model=model):
                    url = self.url(fam, model)
                    m = setup.HF_RESOLVE.match(url)
                    self.assertIsNotNone(m, url)
                    self.assertEqual(m["repo"], fam["hf"].split("/resolve/")[0].split("/", 3)[3])
                    self.assertIn(m["rev"], setup.HF_REVISIONS.values())
                    self.assertEqual(Path(m["name"]).name, fam["file"].format(q=model, i=1))
                    mm = setup.HF_RESOLVE.match(fam["mmproj_hf"] + fam["mmproj"])
                    self.assertIsNotNone(mm, fam["mmproj_hf"])
                    self.assertEqual(mm["name"], fam["mmproj"])
        self.assertEqual(seen, sum(1 for s in setup.MODELS.values()
                                   for f in s.get("families", ("qwen",)) if f in setup.FAMILIES))

    def test_the_coder_model_is_the_one_in_the_cache(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            fam = setup.FAMILIES["coder"]
            url = self.url(fam, "IQ1_M")
            m = setup.HF_RESOLVE.match(url)
            self.assertEqual(m["rev"], setup.HF_REVISIONS["ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"])
            self.assertEqual(m["name"], "IQ1_M/" + fam["file"].format(q="IQ1_M", i=1))
            repo = cache / ("models--" + m["repo"].replace("/", "--"))
            blob = repo / "blobs" / "e11083b"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"29 GB")
            link(repo / "snapshots" / m["rev"] / m["name"], blob)
            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(cache)}):
                dst = cache / "IQ1_M" / Path(url).name
                with mock.patch.object(setup.urllib.request, "urlopen", no_network):
                    with contextlib.redirect_stdout(io.StringIO()):
                        setup.download(url, dst, "Qwen3.8-Flash-Next Coder IQ1_M")
                self.assertEqual(dst.read_bytes(), b"29 GB")
                self.assertTrue(setup.done(dst))
                json.dumps(setup.MODELS)                             # the sizes the user is promised stay untouched


if __name__ == "__main__":
    unittest.main()