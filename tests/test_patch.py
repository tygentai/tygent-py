import asyncio
import sys
import types
import unittest

from tygent.patch import install


class DummyModel:
    async def generate_content(self, prompt):
        return f"patched:{prompt}"


class InstallPatchTest(unittest.TestCase):
    def setUp(self):
        genai = types.ModuleType("google.generativeai")
        genai.GenerativeModel = DummyModel
        google_pkg = types.ModuleType("google")
        google_pkg.generativeai = genai
        sys.modules["google"] = google_pkg
        sys.modules["google.generativeai"] = genai

    def tearDown(self):
        sys.modules.pop("google", None)
        sys.modules.pop("google.generativeai", None)

    def test_install_google_patch(self):
        install()
        from google.generativeai import GenerativeModel

        async def run():
            model = GenerativeModel()
            return await model.generate_content("hi")

        result = asyncio.run(run())
        self.assertEqual(result, "patched:hi")
        self.assertTrue(
            hasattr(GenerativeModel, "_tygent_generate_content")
            or hasattr(GenerativeModel, "_tygent_generateContent")
        )


if __name__ == "__main__":
    unittest.main()
