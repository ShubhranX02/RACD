import re

with open('RACD_Final_Report.md', 'r') as f:
    text = f.read()

# Replace block math
r_orig = r"$$r = -\left( \frac{|\text{Peak}_{\text{sim}} - \text{Peak}_{\text{target}}|}{\Delta \text{Peak}_{\text{span}}} + \max\left(0, \frac{\text{Noise} - 1.5}{1.5}\right) + \max\left(0, \frac{\text{Power} - 15.0}{15.0}\right) \right)$$"
r_new = r"""<div style="text-align:center; margin:20px 0; padding:15px; background:#f0f4f8; border-radius:8px; font-family:'Times New Roman',serif; font-size:18px; font-style:italic;">
r = - ( |Peak<sub>sim</sub> - Peak<sub>target</sub>| / ΔPeak<sub>span</sub> + max(0, (Noise - 1.5)/1.5) + max(0, (Power - 15.0)/15.0) )
</div>"""
text = text.replace(r_orig, r_new)

h_orig = r"$$H(s) = \frac{V_{\text{out}}(s)}{V_{\text{in}}(s)} \approx \frac{1 + s R_s C_s}{1 + s (R_s + R_{\text{load}}) C_s}$$"
h_new = r"""<div style="text-align:center; margin:20px 0; padding:15px; background:#f0f4f8; border-radius:8px; font-family:'Times New Roman',serif; font-size:18px; font-style:italic;">
H(s) = V<sub>out</sub>(s) / V<sub>in</sub>(s) ≈ (1 + s·R<sub>s</sub>·C<sub>s</sub>) / (1 + s·(R<sub>s</sub> + R<sub>load</sub>)·C<sub>s</sub>)
</div>"""
text = text.replace(h_orig, h_new)


# Regex for remaining inline math: replace $...$ with just ...
text = re.sub(r'\$(.*?)\$', lambda m: m.group(1).replace(r'\text{', '').replace('}', '').replace(r'\parallel', '||').replace(r'\times', '×').replace(r'\mu', 'µ').replace(r'\Omega', 'Ω').replace(r'\pm', '±').replace(r'\rightarrow', '→').replace(r'\approx', '≈').replace(r'\cdot', '·').replace('_{', '_').replace('^', ''), text)

with open('RACD_Final_Report.md', 'w') as f:
    f.write(text)
