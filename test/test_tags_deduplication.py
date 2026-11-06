"""
test/test_tags_deduplication.py - Test semantic deduplication and normalization of tags.
"""
import subprocess
import unittest


class TestTagsDeduplication(unittest.TestCase):
    def test_parse_tags_in_node(self):
        js_code = """
        // Read app.js file content and evaluate parseTags
        import fs from 'fs';
        const code = fs.readFileSync('api/static/js/app.js', 'utf-8');
        const start = code.indexOf('parseTags(tags) {');
        const end = code.indexOf('getTagClass(tag) {');
        const block = code.substring(start + 'parseTags(tags) {'.length, end).trim();
        // remove trailing comma and closing brace
        const lastBrace = block.lastIndexOf('}');
        const funcBody = block.substring(0, lastBrace);

        const ctx = {
            stripEmojis: (s) => (s || '').replace(/[\\u{1F300}-\\u{1F9FF}]/gu, '').trim()
        };
        const parseTags = new Function('tags', funcBody).bind(ctx);

        const res1 = parseTags('TimesFM看多 | TimesFM看漲 | 季線支撐');
        const labels1 = res1.map(r => r.label);
        if (res1.filter(r => r.label.includes('TimesFM')).length !== 1) {
            throw new Error('TimesFM not deduplicated: ' + JSON.stringify(labels1));
        }

        const res2 = parseTags('MOIRAI2看漲 | MOIRAI-2看多 | 主流板塊(AI) | 主流板塊(散熱)');
        const labels2 = res2.map(r => r.label);
        if (res2.filter(r => r.label.includes('MOIRAI-2')).length !== 1) {
            throw new Error('MOIRAI-2 not deduplicated: ' + JSON.stringify(labels2));
        }
        if (res2.filter(r => r.label.includes('主流板塊')).length !== 1) {
            throw new Error('主流板塊 not deduplicated: ' + JSON.stringify(labels2));
        }

        const res3 = parseTags('高盈虧比 | 風報比優 | 偏多看好 | 偏多佈局');
        const labels3 = res3.map(r => r.label);
        if (res3.filter(r => r.label.includes('高盈虧比')).length !== 1) {
            throw new Error('高盈虧比 not deduplicated: ' + JSON.stringify(labels3));
        }
        if (res3.filter(r => r.label.includes('偏多')).length !== 1) {
            throw new Error('Stance not deduplicated: ' + JSON.stringify(labels3));
        }

        const res4 = parseTags('TimesFM看多 | 多方看漲 | 看多');
        const labels4 = res4.map(r => r.label);
        if (labels4.includes('多方看漲') || labels4.includes('看多')) {
            throw new Error('Generic bullish not filtered when specific model bullish exists: ' + JSON.stringify(labels4));
        }
        if (!labels4.includes('TimesFM看多')) {
            throw new Error('TimesFM看多 should be kept: ' + JSON.stringify(labels4));
        }

        const res5 = parseTags('土洋合買 | 法人籌碼 | 法人買超');
        const labels5 = res5.map(r => r.label);
        if (labels5.includes('法人籌碼') || labels5.includes('法人買超')) {
            throw new Error('Generic institutional tag not filtered when 土洋合買 exists: ' + JSON.stringify(labels5));
        }
        if (!labels5.includes('土洋合買')) {
            throw new Error('土洋合買 should be kept: ' + JSON.stringify(labels5));
        }

        console.log('SUCCESS');
        """
        proc = subprocess.run(
            ["node", "--input-type=module", "-e", js_code],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc.returncode, 0, msg=f"Node error: {proc.stderr}\n{proc.stdout}")
        self.assertIn("SUCCESS", proc.stdout)


if __name__ == "__main__":
    unittest.main()
