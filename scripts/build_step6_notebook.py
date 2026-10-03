"""Create and execute an inspection notebook using this exact Python runtime."""
from pathlib import Path
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook()
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
nb.cells=[
md('# Step 6: restricted-EWL ESS resolution\n\nAll four survivors are certified within the frozen restricted family. Read `docs/step6_methods.md` for the complete argument. This notebook inspects saved data and rederives the exact certificate; it does not treat numerical near-zero as mathematical equality. Steps 7-10 are not executed.'),
code("from pathlib import Path\nimport sys, json, gzip\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nif not (ROOT/'src/quantum_ess').exists(): ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT/'src'))\nDATA = ROOT/'results/step6'\nmeta = json.loads((DATA/'run_metadata.json').read_text())\ndisplay({k: meta[k] for k in ['git_head_at_run','step5_merge_commit','python','hardware','runtime_seconds','seeds','workers','tolerance']})"),
md('## Four residents\nThe reported strongest candidates below are finite-grid maxima. The continuous near-self supremum has no distinct maximizing mutant.'),
code("summary = pd.read_csv(DATA/'resident_summary.csv')\ndisplay(summary[['resident','resident_theta','resident_phi','resident_payoff','step5_status','step6_status','mutant_theta','mutant_phi','distance','leading_order','leading_coefficient','power_coefficients','delta_001']])"),
md('## Exact identities and complete tie set\nLet x=cos(theta/2)cos(phi), y=cos(theta/2)sin(phi), s=sin(theta/2), with x,y,s>=0 and x²+y²+s²=1. Q and A have strictly negative a0 away from self. B has a0=-(9/2)sx, with a1=-3/4 on s=0 and a1=-(3/4)(2y²-1)² on x=0. Equality at the second stage occurs only at self B.'),
code("from quantum_ess.ewl_resolution import symbolic_certificate\ncert = symbolic_certificate()\nassert cert == json.loads((DATA/'certification_summary.json').read_text())\ndisplay(cert)\ndisplay(json.loads((DATA/'neutral_sets.json').read_text()))"),
md('## Convergence\nSelf-excluded canonical counts are 220, 840 and 3280 per resident. These grids support the exact proof but are not a proof themselves.'),
code("convergence=pd.read_csv(DATA/'grid_convergence.csv')\ndisplay(convergence)\ndisplay(Image(filename=str(DATA/'plots/grid_convergence.png')))"),
md('## Local geometry and boundary handling\nB has a nonzero normal gradient and a zero a0 Hessian. Its tangent direction is controlled by the negative quadratic a1. The other three a0 Hessians are negative definite.'),
code("local=json.loads((DATA/'local_derivative_analysis.json').read_text())\ndisplay({name: {'cone':v['feasible_cone'],'a0':v['coefficients'][0], 'null_resolution':v['null_direction_resolution']} for name,v in local.items()})\ndisplay(Image(filename=str(DATA/'plots/B_tie_set.png')))"),
md('## Precision and optimizer transparency\nTiny numerical values are preserved as strings. There are 52 converged global searches and one failed local line search out of 32 local refinements. The proof does not depend on these optimizer outcomes.'),
code("precision=pd.read_csv(DATA/'precision_validation.csv',dtype=str)\ndisplay(precision[['resident','probe','high_precision_leading_order','high_precision_leading_value','max_double_error','ordinary_coordinates_collapse_to_self']])\noptimizers=pd.read_csv(DATA/'coefficient_optimizer_runs.csv')\ndisplay(optimizers.groupby(['method','success']).size())\ndisplay(optimizers[~optimizers.success])"),
md('## Raw data and reproducible validator'),
code("raw=pd.read_csv(DATA/'mutant_coefficients.csv.gz')\nassert len(raw)==17360\ndisplay(raw.groupby(['resident','stage','numerical_classification']).size())\ndisplay(raw[['resident','stage','u_R','u_M','bernstein_differences','power_coefficients']].head())\nimport importlib.util\nspec=importlib.util.spec_from_file_location('validate_step6',ROOT/'scripts/validate_step6_results.py')\nvalidator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)\ndisplay(validator.validate(DATA))"),
md('## Next research step\nReview the exact identities and zero sets. Step 7, only after authorization, adds full SU(2) and starts with a unilateral Nash screen. Restricted-family stability does not transfer automatically. The report includes the stored Iqbal-Toor and Benjamin-Hayden references and the future Steps 8-10 roadmap.')]
nb.metadata={'kernelspec':{'display_name':'Python 3 (Step 6 research)','language':'python','name':'python3'},'language_info':{'name':'python','version':sys.version.split()[0]}}
km=KernelManager(kernel_name='python3');km.kernel_spec.argv[0]=sys.executable
client=NotebookClient(nb,km=km,timeout=120,resources={'metadata':{'path':str(ROOT)}})
try:
    client.execute()
finally:
    if km.has_kernel:km.shutdown_kernel(now=True)
path=ROOT/'notebooks/step6_restricted_ewl_ess_resolution.ipynb'
nbf.validate(nb);nbf.write(nb,path)
assert not any(output.get('output_type')=='error' for cell in nb.cells if cell.cell_type=='code' for output in cell.outputs)
print(path)
