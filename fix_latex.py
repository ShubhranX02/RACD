import re

with open('RACD_Final_Report.md', 'r') as f:
    text = f.read()

# Replace vector math
v_orig = r"$$\mathbf{v} = \left[ \frac{\text{Peak}}{11.0}, \, \frac{\text{Noise}}{1.5}, \, \frac{\text{Eye}}{700.0}, \, \frac{R_s - 50}{450}, \, \frac{C_s - 0.1\,\text{pF}}{9.9\,\text{pF}} \right]$$"
v_new = r"""<div style="text-align:center; margin:20px 0; padding:15px; background:#f0f4f8; border-radius:8px; font-family:'Times New Roman',serif; font-size:18px; font-style:italic;">
v = [ Peak/11.0 , Noise/1.5 , Eye/700.0 , (Rs-50)/450 , (Cs-0.1pF)/9.9pF ]
</div>"""
text = text.replace(v_orig, v_new)

# Replace BC math
bc_orig = r"$$\mathcal{L}_{\text{BC}}(\theta) = \frac{1}{N} \sum_{i=1}^N \left\| \pi_\theta(s_i) - a_i^* \right\|_2^2$$"
bc_new = r"""<div style="text-align:center; margin:20px 0; padding:15px; background:#f0f4f8; border-radius:8px; font-family:'Times New Roman',serif; font-size:18px; font-style:italic;">
L<sub>BC</sub>(θ) = (1/N) Σ || π<sub>θ</sub>(s<sub>i</sub>) - a<sub>i</sub><sup>*</sup> ||<sup>2</sup>
</div>"""
text = text.replace(bc_orig, bc_new)

# Replace inline ratio math
text = text.replace(r"$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$", "r<sub>t</sub>(θ) = π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) / π<sub>θ_old</sub>(a<sub>t</sub>|s<sub>t</sub>)")
text = text.replace(r"$\hat{A}_t$", "Â<sub>t</sub>")

# Replace remaining $...$
replacements = {
    r"$\mathcal{S}$": "S",
    r"$\mathcal{A}$": "A",
    r"$\mathcal{R}$": "R",
    r"$[-1, 1]$": "[-1, 1]",
    r"$W_{n1}$": "Wn1",
    r"$L = 0.15\,\mu\text{m}$": "L = 0.15 µm",
    r"$R_{s1}, C_{s1}$": "Rs1, Cs1",
    r"$750.0\,\Omega \parallel 1.80\,\text{pF}$": "750.0 Ω || 1.80 pF",
    r"$R_{L1}, I_{\text{tail1}}$": "RL1, Itail1",
    r"$2200.0\,\Omega$": "2200.0 Ω",
    r"$400.0\,\mu\text{A}$": "400.0 µA",
    r"$800\,\mu\text{A}$": "800 µA",
    r"$C_{ac} = 5.0\,\text{pF}$": "Cac = 5.0 pF",
    r"$V_{\text{mid}} = V_{\text{dd}}/2$": "Vmid = Vdd/2",
    r"$50\,\text{k}\Omega$": "50 kΩ",
    r"$W_{n2}$": "Wn2",
    r"$R_{s2}, C_{s2}$": "Rs2, Cs2",
    r"$R_{L2}, I_{\text{tail2}}$": "RL2, Itail2",
    r"$R_{\text{dfe}}$": "Rdfe",
    r"$20.0\,\text{k}\Omega$": "20.0 kΩ",
    r"$200\,\text{ps}$": "200 ps",
    r"$\mathbf{0.01951\,\text{mm}^2}$": "**0.01951 mm²**",
    r"$< 0.050\,\text{mm}^2$": "< 0.050 mm²",
    r"$< -30$ dB": "< -30 dB",
    r"$< 1.5$ mV": "< 1.5 mV",
    r"$< 15$ mW": "< 15 mW",
    r"$> 100$ mV": "> 100 mV",
    r"$< 0.05 \text{mm}^2$": "< 0.05 mm²",
    r"$A_1(s) \times A_2(s)$": "A1(s) × A2(s)",
    r"$5.27 \sim 5.90$ dB": "5.27 ~ 5.90 dB",
    r"$1 + g_m R_s / 2 \approx 1.5$": "1 + gm·Rs/2 ≈ 1.5",
    r"$\approx 2.9\text{ dB}$": "≈ 2.9 dB",
    r"$|H_{\text{tot}}(s)| = |H_1(s)| \cdot |H_2(s)|$": "|H_tot(s)| = |H1(s)| · |H2(s)|",
    r"$\text{HD3} = -32.0 \sim -37.2\text{ dB}$": "HD3 = -32.0 ~ -37.2 dB",
    r"$\text{HD3}$": "HD3",
    r"$< -30.0\text{ dB}$": "< -30.0 dB",
    r"$< 15.0\text{ mW}$": "< 15.0 mW",
    r"$0.01951\text{ mm}^2$": "0.01951 mm²",
    r"$< 0.050\text{ mm}^2$": "< 0.050 mm²",
    r"$4.00\,\mu\text{m}$": "4.00 µm",
    r"$3.0 \sim 12.0\text{ dB}$": "3.0 ~ 12.0 dB",
    r"$HD_3 < -35$ dB": "HD3 < -35 dB",
    r"$< -30$": "< -30"
}

for k, v in replacements.items():
    text = text.replace(k, v)

with open('RACD_Final_Report.md', 'w') as f:
    f.write(text)

