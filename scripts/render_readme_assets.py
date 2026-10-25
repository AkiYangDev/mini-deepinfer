"""Regenerate the README's small SVG assets using only the standard library."""

from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT/'docs/assets'


def write(name, svg):
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS/name).write_bytes((svg.strip()+'\n').encode('utf-8'))


def main():
    write('readme-hero.svg', '''
<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="300" viewBox="0 0 1280 300" role="img" aria-labelledby="title desc">
  <title id="title">mini-DeepInfer</title>
  <desc id="desc">DeepInfer design. Real models. Ascend 910C. CPU walkthroughs, Qwen3.5 and speculative decoding. The state blocks are a brand illustration, not a hardware or timing diagram.</desc>
  <defs>
    <linearGradient id="bg" x2="1" y2="1"><stop stop-color="#0b1323"/><stop offset="1" stop-color="#142941"/></linearGradient>
    <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M 32 0 H 0 V 32" fill="none" stroke="#a1c8ed" stroke-opacity=".055"/></pattern>
  </defs>
  <rect width="1280" height="300" rx="20" fill="url(#bg)"/>
  <rect width="1280" height="300" rx="20" fill="url(#grid)"/>
  <path d="M 0 32 V 268 Q 0 300 32 300 H 1248" fill="none" stroke="#38bdf8" stroke-opacity=".3"/>
  <g font-family="Arial, Helvetica, sans-serif">
    <text x="56" y="59" fill="#86cdec" font-size="15" font-weight="700" letter-spacing="3">INFERENCE, STEP BY STEP</text>
    <text x="52" y="135" font-size="68" font-weight="700" letter-spacing="-2"><tspan fill="#93b2cf">mini-</tspan><tspan fill="#f0f7ff">DeepInfer</tspan></text>
    <text x="56" y="179" fill="#c2d3e4" font-size="24">DeepInfer design. Real models. Ascend 910C.</text>
    <g font-size="16" font-weight="700">
      <rect x="56" y="219" width="149" height="36" rx="18" fill="#17334a"/><text x="77" y="243" fill="#76d5fa">CPU to NPU</text>
      <rect x="217" y="219" width="145" height="36" rx="18" fill="#263642"/><text x="248" y="243" fill="#fdba74">Qwen3.5</text>
      <rect x="374" y="219" width="209" height="36" rx="18" fill="#163e42"/><text x="395" y="243" fill="#79dfcb">Speculative decoding</text>
    </g>
    <g transform="translate(927 39)">
      <rect width="295" height="222" rx="17" fill="#0c1a2b" stroke="#355878"/>
      <text x="24" y="36" fill="#9bb8d1" font-size="12" letter-spacing="2">EXPLICIT STATE</text>
      <circle cx="264" cy="31" r="4" fill="#5eead4"/>
      <g font-family="Consolas, monospace" font-size="18">
        <text x="24" y="77" fill="#7dd3fc">&gt; prefill</text>
        <text x="24" y="111" fill="#cbdcec">&gt; decode</text>
        <text x="24" y="145" fill="#5eead4">&gt; commit</text>
      </g>
      <path d="M 229 63 H 252 V 130 H 230" fill="none" stroke="#52718e" stroke-width="2"/>
      <path d="M 235 124 L 229 130 L 235 136" fill="none" stroke="#52718e" stroke-width="2"/>
      <g fill="#234562"><rect x="24" y="179" width="29" height="14" rx="3"/><rect x="61" y="179" width="29" height="14" rx="3"/><rect x="98" y="179" width="29" height="14" rx="3"/></g>
      <g fill="#317c8b"><rect x="135" y="179" width="29" height="14" rx="3"/><rect x="172" y="179" width="29" height="14" rx="3"/></g>
      <rect x="209" y="179" width="29" height="14" rx="3" fill="#e8a66a"/>
      <rect x="246" y="179" width="25" height="14" rx="3" fill="#234562"/>
    </g>
  </g>
</svg>''')
    transcript = (ASSETS/'pipeline-demo.txt').read_text(encoding='utf-8')
    lines = (
        '  B: finished; cached CPU/issued=3/3; slots=0',
        '  C: running; cached CPU/issued=0/3; slots=6',
        '  reused B -> C: slots [14, 15, 16]',
        '  free slots: 0/20',
    )
    for line in lines:
        if line not in transcript.splitlines():
            raise ValueError('terminal card must use preserved output: '+line)
    rows = '\n'.join(f'<text x="30" y="{91+index*34}" fill="{color}" font-size="20">{escape(line)}</text>'
                     for index,(line,color) in enumerate(zip(lines,('#a4b8cd','#bce7fa','#6ee7cf','#c7d7e7'))))
    write('cache-reuse.svg', f'''
<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="264" viewBox="0 0 1000 264" role="img" aria-labelledby="title desc">
  <title id="title">A real CPU demo: B releases its slots and C reuses them</title>
  <desc id="desc">Excerpt from python -m minideepinfer.pipeline_demo. B has finished, C owns slots 14, 15, 16, and no slots are free because they are already reused.</desc>
  <rect x="1" y="1" width="998" height="262" rx="14" fill="#101b2b" stroke="#30465e"/>
  <path d="M 1 48 H 999" stroke="#30465e"/>
  <circle cx="24" cy="25" r="5" fill="#f18d8d"/><circle cx="43" cy="25" r="5" fill="#e5ba70"/><circle cx="62" cy="25" r="5" fill="#79c9ab"/>
  <text x="88" y="31" font-family="Consolas, monospace" font-size="16" fill="#8ea6bf">python -m minideepinfer.pipeline_demo</text>
  <g font-family="Consolas, monospace" xml:space="preserve">{rows}</g>
  <text x="43" y="238" font-family="Arial, Helvetica, sans-serif" font-size="18" fill="#77d9c3">B released its cache. C already owns the slots.</text>
</svg>''')


if __name__ == '__main__':
    main()
