"""
orchestrator.py — Natural Language to Production Transistor Netlist Engine.

Converts plain English engineering design requests into production-grade SkyWater 130 nm
equalizer netlists and verified simulation datasheets using the Hybrid RACD (FAISS + RL) engine.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(__file__))

from circuit_transistor import _device_section_1stage, PDK_LIB_PATH
from environment_transistor import estimate_area_1stage_mm2

# Safe Anthropic initialization with offline regex fallback
client = None
try:
    import anthropic
    if os.environ.get("ANTHROPIC_API_KEY"):
        client = anthropic.Anthropic()
except Exception:
    client = None

SYSTEM_CONTEXT = """You are a specification parser for an analog equalizer design tool.
The tool designs a single-stage CTLE + 1-Tap DFE in SkyWater 130 nm PDK for high-speed SerDes (PCIe Gen2 / 5 Gbps).
Extract the following engineering targets from the user request:
- target_peaking_db: High-frequency peaking boost, must be between 3.0 and 12.0 dB (default 8.0 dB).
- noise_limit_mvrms: Maximum acceptable input-referred noise (default 1.5 mVrms).
- power_limit_mw: Maximum allowable power consumption (default 15.0 mW).
- eye_height_limit_mv: Minimum eye opening (default 100.0 mV).

Users describe what they want in plain language (e.g. "8.5 dB high peaking, keep noise low and under 1.2 mV").
Return ONLY a valid JSON object with keys: target_peaking_db, noise_limit_mvrms, power_limit_mw, eye_height_limit_mv.
"""


def parse_spec_request(user_text: str) -> Dict[str, float]:
    """Converts natural language request into structured target specifications."""
    if client is not None:
        try:
            prompt = f"{SYSTEM_CONTEXT}\n\nRequest: \"{user_text}\"\nReturn ONLY JSON."
            response = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Strip any markdown fences
            text = re.sub(r'```(?:json)?', '', text).strip()
            spec = json.loads(text)
            return {
                "target_peaking_db": max(3.0, min(12.0, float(spec.get("target_peaking_db", 8.0)))),
                "noise_limit_mvrms": float(spec.get("noise_limit_mvrms", 1.5)),
                "power_limit_mw": float(spec.get("power_limit_mw", 15.0)),
                "eye_height_limit_mv": float(spec.get("eye_height_limit_mv", 100.0)),
            }
        except Exception:
            pass

    # High-accuracy regex extraction fallback (offline / demo mode)
    m_peak = re.search(r'(\d+\.?\d*)\s*(?:db|peaking|boost)', user_text, re.IGNORECASE)
    if not m_peak:
        m_peak = re.search(r'(?:peaking|boost|around)\s*(\d+\.?\d*)', user_text, re.IGNORECASE)
    target_peaking = float(m_peak.group(1)) if m_peak else 8.0
    target_peaking = max(3.0, min(12.0, target_peaking))

    m_noise = re.search(r'(?:noise|strict)\s*(?:under|below|of)?\s*(\d+\.?\d*)', user_text, re.IGNORECASE)
    noise = float(m_noise.group(1)) if m_noise else 1.5

    m_power = re.search(r'(?:power)\s*(?:under|below|of)?\s*(\d+\.?\d*)', user_text, re.IGNORECASE)
    power = float(m_power.group(1)) if m_power else 15.0

    return {
        "target_peaking_db": target_peaking,
        "noise_limit_mvrms": noise,
        "power_limit_mw": power,
        "eye_height_limit_mv": 100.0,
    }


def generate_spice_netlist(
    sizing: Dict[str, float],
    corner: str = "tt",
    temp: int = 27,
    vdd: float = 1.8,
) -> str:
    """Generate a production-ready, stand-alone ngspice netlist for the synthesized 1-stage CTLE."""
    Wn = sizing["Wn_um"]
    Rs = sizing["Rs_ohm"]
    Cs = sizing["Cs_farad"]
    Itail = sizing["Itail_half_ua"]
    RL = sizing["RL_ohm"]
    Rdfe = sizing["Rdfe_ohm"]

    devices = _device_section_1stage(Wn, Rs, Cs, Itail, RL, Rdfe)

    netlist = f"""* RACD Synthesized SkyWater 130 nm 1-Stage CTLE + 1-Tap DFE
* Technology: SkyWater 130 nm PDK (sky130A)
* Synthesis Sizing: Wn={Wn:.3f}um, Rs={Rs:.1f} Ohm, Cs={Cs*1e12:.3f}pF, Itail={Itail:.1f}uA, RL={RL:.1f} Ohm, Rdfe={Rdfe:.1f} Ohm

.lib "{PDK_LIB_PATH}" {corner}
.option temp={temp} scale=1u

Vdd vdd 0 {vdd}
Vinp vin_p 0 DC {vdd/2} AC 0.5
Vinn vin_n 0 DC {vdd/2} AC -0.5

{devices}

.control
ac dec 20 10meg 10g
wrdata ctle_ac_response.data vdb(voutp,voutn)
op
print i(Vdd)
noise v(voutp,voutn) vinp dec 20 10meg 5g
print inoise_total
.endc
.end
"""
    return netlist


def synthesize_from_prompt(
    user_prompt: str,
    run_spice_validation: bool = True,
    output_netlist_path: Optional[str] = None,
) -> Dict:
    """
    End-to-end pipeline: Prompt -> Structured Specs -> RACD Synthesis -> Netlist -> SPICE Audit.
    """
    print("=" * 80)
    print(f"  RACD Natural Language Synthesis Engine")
    print(f"  Prompt: \"{user_prompt}\"")
    print("=" * 80)

    # 1. Parse Specs
    specs = parse_spec_request(user_prompt)
    print(f"[1/4] Extracted Specifications:")
    print(f"      Target Peaking:    {specs['target_peaking_db']:.2f} dB")
    print(f"      Noise Limit:       {specs['noise_limit_mvrms']:.2f} mVrms")
    print(f"      Power Budget:      {specs['power_limit_mw']:.2f} mW")
    print(f"      Min Eye Opening:   {specs['eye_height_limit_mv']:.1f} mV")

    # 2. Hybrid RACD Synthesis
    print(f"[2/4] Querying Transistor RACD (FAISS + RL)...")
    from retrieval_faiss_transistor import synthesize_transistor_racd
    res = synthesize_transistor_racd(
        target_peaking_db=specs["target_peaking_db"],
        noise_limit_mvrms=specs["noise_limit_mvrms"],
        power_limit_mw=specs["power_limit_mw"],
        eye_height_limit_mv=specs["eye_height_limit_mv"],
        corner="tt",
    )
    sizing = res["sizing"]
    area = estimate_area_1stage_mm2(
        sizing["Wn_um"], sizing["Rs_ohm"], sizing["Cs_farad"], sizing["RL_ohm"], sizing["Rdfe_ohm"]
    )

    print(f"      Synthesis Source:  {res['source']}")
    print(f"      Query Latency:     {res['elapsed_ms']:.2f} ms")
    print(f"      Synthesized Parameters:")
    print(f"        Wn (Differential Pair): {sizing['Wn_um']:.3f} um")
    print(f"        Rs (Degeneration Res):  {sizing['Rs_ohm']:.1f} Ohm")
    print(f"        Cs (Degeneration Cap):  {sizing['Cs_farad']*1e12:.3f} pF")
    print(f"        Itail (Tail Current):   {sizing['Itail_half_ua']:.1f} uA")
    print(f"        RL (Load Resistor):     {sizing['RL_ohm']:.1f} Ohm")
    print(f"        Rdfe (DFE Tap Weight):  {sizing['Rdfe_ohm']:.1f} Ohm")
    print(f"        Die Area Estimate:      {area:.5f} mm2")

    # 3. Export Standalone Netlist
    print(f"[3/4] Generating Production SPICE Netlist...")
    netlist_str = generate_spice_netlist(sizing, corner="tt", temp=27, vdd=1.8)
    if output_netlist_path is None:
        netlist_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        os.makedirs(netlist_dir, exist_ok=True)
        output_netlist_path = os.path.join(netlist_dir, "synthesized_ctle.cir")

    with open(output_netlist_path, "w", encoding="utf-8") as f:
        f.write(netlist_str)
    print(f"      Saved Netlist: {output_netlist_path}")

    # 4. Optional Ground-Truth SPICE Sign-off
    spice_metrics = None
    if run_spice_validation:
        print(f"[4/4] Executing Ground-Truth SPICE Sign-Off...")
        from circuit_transistor import simulate_transistor_fast
        spice_res = simulate_transistor_fast(
            sizing["Wn_um"], sizing["Rs_ohm"], sizing["Cs_farad"],
            sizing["Itail_half_ua"], sizing["RL_ohm"], sizing["Rdfe_ohm"],
            corner="tt", temp=27, vdd=1.8, topology="1stage"
        )
        spice_metrics = spice_res
        err = abs(spice_res["peaking_db"] - specs["target_peaking_db"])
        print(f"      Achieved Peaking:  {spice_res['peaking_db']:.2f} dB (Error: {err:.2f} dB)")
        print(f"      Noise (Input-Ref): {spice_res['noise_mvrms']:.3f} mVrms (Spec: <= {specs['noise_limit_mvrms']:.2f} mV)")
        print(f"      Power Consumption: {spice_res['power_mw']:.2f} mW (Spec: <= {specs['power_limit_mw']:.2f} mW)")
        print(f"      Eye Opening Proxy: {spice_res['eye_height_proxy_mv']:.1f} mV")
    else:
        print(f"[4/4] Skipped SPICE sign-off.")

    print("=" * 80)
    if spice_metrics is not None:
        # Evaluate compliance against extracted specs
        err_db = abs(spice_metrics["peaking_db"] - specs["target_peaking_db"])
        noise_ok  = spice_metrics["noise_mvrms"]  <= specs["noise_limit_mvrms"]
        power_ok  = spice_metrics["power_mw"]    <= specs["power_limit_mw"]
        eye_ok    = spice_metrics["eye_height_proxy_mv"] >= specs["eye_height_limit_mv"]
        peaking_ok = err_db <= 1.5
        all_ok = peaking_ok and noise_ok and power_ok and eye_ok
        if all_ok:
            print("  SYNTHESIS COMPLETE — ALL SPECS MET ✓")
        else:
            failures = []
            if not peaking_ok:  failures.append(f"Peaking error={err_db:.2f}dB")
            if not noise_ok:    failures.append(f"Noise={spice_metrics['noise_mvrms']:.3f}mVrms")
            if not power_ok:    failures.append(f"Power={spice_metrics['power_mw']:.2f}mW")
            if not eye_ok:      failures.append(f"Eye={spice_metrics['eye_height_proxy_mv']:.1f}mV")
            print(f"  SYNTHESIS COMPLETE — MARGINAL: {', '.join(failures)}")
    else:
        print("  SYNTHESIS COMPLETE — SPICE validation skipped")
    print("=" * 80)

    return {
        "prompt": user_prompt,
        "specs": specs,
        "sizing": sizing,
        "area_mm2": area,
        "netlist_path": output_netlist_path,
        "spice_metrics": spice_metrics,
        "racd_source": res["source"],
    }


if __name__ == "__main__":
    test_prompt = "Design a 2-stage CTLE for PCIe Gen2 with 8.0 dB peaking boost, strict noise under 1.2 mV, and low power"
    synthesize_from_prompt(test_prompt, run_spice_validation=True)
