"""Bundle the verified static app into an offline, single-file HTML artifact."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'web/explainer'
html=(SOURCE/'index.html').read_text();css='\n'.join(line for line in (SOURCE/'style.css').read_text().splitlines() if not line.startswith('@import'))
html=html.replace('<link rel="stylesheet" href="style.css">','<style>'+css+'</style>').replace('<script defer src="data.js"></script>','').replace('<script defer src="app.js"></script>','')
html=html.replace('</body>','<script>'+(SOURCE/'data.js').read_text()+'\n'+(SOURCE/'app.js').read_text()+'</script></body>')
out=ROOT/'output/web/WSP4Rec_interactive.html';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(html);print(out)
