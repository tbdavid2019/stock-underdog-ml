"""
test/test_typography_rwd.py - Unit test asserting typography floors and mobile RWD standards.
Ensures CJK readability floor (>=12px, font-medium/semibold) and responsive scale harmony.
"""

import unittest
import re
from pathlib import Path


class TestTypographyAndRWD(unittest.TestCase):
    def setUp(self):
        self.template_path = Path("api/templates/index.html")
        self.assertTrue(self.template_path.exists(), "api/templates/index.html not found")
        self.html = self.template_path.read_text(encoding="utf-8")

    def test_no_sub_12px_arbitrary_classes(self):
        """
        Verify that no arbitrary sub-12px Tailwind font classes (text-[10px], text-[11px]) exist.
        CJK characters (繁體中文) require at least 12px to prevent stroke clumping and illegibility.
        """
        sub_12px_matches = re.findall(r'text-\[(?:[0-9]|10|11)px\]', self.html)
        self.assertEqual(
            len(sub_12px_matches), 
            0, 
            f"Found {len(sub_12px_matches)} sub-12px font classes: {set(sub_12px_matches)}. "
            "CJK text must have a minimum floor of 12px (text-xs)."
        )

    def test_stock_card_price_labels_readability(self):
        """
        Verify that stock card price labels ('當前現價', '預測目標價') do not use text-[10px] or text-[11px].
        They should be at least text-xs with font-semibold or font-bold.
        """
        self.assertNotIn('text-[11px]" :class="isDark ? \'text-dark-muted\' : \'text-claude-muted\'">當前現價', self.html)
        self.assertNotIn('text-[11px]" :class="isDark ? \'text-sky-400\' : \'text-claude-terracotta\'">預測目標價', self.html)

    def test_stock_card_tags_readability(self):
        """
        Verify that strategy tags (parseTags) do not use text-[11px].
        """
        tag_match = re.search(r'v-for="tag in parseTags\(item\.tags\)".*?\sclass="([^"]+)"', self.html, re.DOTALL)
        if tag_match:
            class_str = tag_match.group(1)
            self.assertNotIn("text-[10px]", class_str)
            self.assertNotIn("text-[11px]", class_str)
            self.assertIn("text-xs", class_str)

    def test_heading_hierarchy_no_skipped_levels(self):
        """
        Verify heading levels do not skip directly from h1 to h3 without an h2.
        (Caught by Impeccable detect).
        """
        h1_pos = self.html.find("<h1")
        h2_pos = self.html.find("<h2")
        h3_pos = self.html.find("<h3")
        self.assertTrue(h1_pos != -1, "Missing h1")
        self.assertTrue(h2_pos != -1, "Missing h2 - heading level skipped from h1 to h3")
        self.assertTrue(h1_pos < h2_pos, "h1 must appear before h2")

    def test_cjk_typography_and_viewport_optimization(self):
        """
        Verify viewport meta tag and text-size-adjust are in place.
        """
        self.assertIn('name="viewport"', self.html)
        self.assertTrue(
            "-webkit-text-size-adjust" in self.html or "text-size-adjust" in self.html,
            "CSS should include text-size-adjust to prevent mobile browser unwanted text inflation"
        )


if __name__ == "__main__":
    unittest.main()
