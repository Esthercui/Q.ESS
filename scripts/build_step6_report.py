"""Create scientific plots and the human-readable Step 6 Markdown/PDF report."""
from pathlib import Path
import csv,json,math,re
from html import escape
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,Preformatted,KeepTogether
from quantum_ess.ewl_resolution import RESIDENTS,exact_a0,tie_a1

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/step6';OUT=ROOT/'output/step6';PLOTS=DATA/'plots'


def csv_rows(name):
    with (DATA/name).open() as f:return list(csv.DictReader(f))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def build():
    OUT.mkdir(parents=True,exist_ok=True);PLOTS.mkdir(exist_ok=True)
    meta=json.loads((DATA/'run_metadata.json').read_text());summary=csv_rows('resident_summary.csv');conv=csv_rows('grid_convergence.csv');prec=csv_rows('precision_validation.csv');opt=csv_rows('coefficient_optimizer_runs.csv')
    validation=json.loads((DATA/'validation_summary.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    palette=['#156b8a','#c46b2a','#714a9a','#308163']
    def save(fig,name):
        fig.savefig(PLOTS/(name+'.png'),dpi=190,bbox_inches='tight');fig.savefig(PLOTS/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.3),layout='constrained')
    for name,color in zip(RESIDENTS,palette):
        rs=[r for r in conv if r['resident']==name]
        ax.loglog([int(r['distinct_operations']) for r in rs],[-float(r['maximum_delta_001']) for r in rs],'o-',color=color,label=name)
    ax.set(xlabel='Distinct grid operations (including self)',ylabel='Negative of maximum DeltaPi(0.01)',title='Refinement finds closer mutants; every grid maximum stays negative')
    ax.legend(ncol=2);ax.grid(alpha=.2);save(fig,'grid_convergence')
    t=np.linspace(0,math.pi,201);p=np.linspace(0,math.pi/2,151);tt,pp=np.meshgrid(t,p)
    fig,axes=plt.subplots(2,2,figsize=(9,6.5),layout='constrained')
    for ax,name in zip(axes.flat,RESIDENTS):
        z=exact_a0(name,tt,pp);im=ax.pcolormesh(tt/math.pi,pp/math.pi,z,shading='auto',cmap='viridis')
        _,rt,rp,_=RESIDENTS[name];ax.scatter([rt/math.pi],[rp/math.pi],s=35,c='red',marker='x')
        ax.set(title=name+' : exact a0',xlabel='theta / pi',ylabel='phi / pi');fig.colorbar(im,ax=ax,shrink=.8)
    save(fig,'a0_landscapes')
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    axes[0].plot(t/math.pi,tie_a1('phi_pi_over_2',t),color=palette[2]);axes[0].scatter([.5],[0],color='red',label='self B');axes[0].legend()
    axes[0].set(title='B: phi=pi/2 tie edge',xlabel='theta / pi',ylabel='a1 = -3/4 cos(theta)^2')
    axes[1].plot(p/math.pi,tie_a1('theta_0',p),color=palette[2]);axes[1].set(ylim=(-.85,-.65),title='B: theta=0 tie edge',xlabel='phi / pi',ylabel='a1 = -3/4')
    for ax in axes:ax.grid(alpha=.2)
    save(fig,'B_tie_set')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    h=np.logspace(-20,-1,120)
    for name,color in zip(RESIDENTS,palette):
        if name=='K4_B':values=.75*np.sin(h)**2;label=name+' (-a1 on exact tie)'
        else:
            alpha=3 if name=='K2_Q' else 9 if name=='K4_Q' else (9+5*math.sqrt(5))/2
            values=alpha*np.sin(h/2)**2;label=name+' (-a0)'
        ax.loglog(h,values,color=color,label=label)
    ax.axhline(1e-8,ls='--',color='grey',lw=1,label='Step 5 coefficient threshold')
    ax.set(xlabel='Feasible theta displacement from resident',ylabel='Magnitude of first nonzero coefficient',title='True rejection can lie far below the floating classification threshold')
    ax.legend(fontsize=8);ax.grid(alpha=.2);save(fig,'near_self_scaling')
    proof=[('K2_Q','-3 sin(theta/2)^2 - 2 cos(theta/2)^2 cos(phi)^2','Only self ties; a0<0 for every distinct mutant.'),
           ('K4_Q','-9 sin(theta/2)^2 - 6 cos(theta/2)^2 cos(phi)^2','Only self ties; a0<0 for every distinct mutant.'),
           ('K4_B','-(9/4) sin(theta) cos(phi)','On theta=0: a1=-3/4. On phi=pi/2: a1=-(3/4)cos(theta)^2; equality only at B.'),
           ('K5_A','-alpha sin(theta/2)^2 - 8 cos(theta/2)^2 sin(phi-2pi/5)^2','alpha=(9+5sqrt(5))/2>0. Only self ties.')]
    text=['# Step 6: Restricted EWL ESS Resolution',
          'Q-ESS research handoff | 2 October 2026 (America/Phoenix) | Version 1',
          '## Result and scope',
          '**All four Step 5 survivors are proven ESS within the specified restricted two-parameter EWL family.** The proof uses exact coefficient identities and complete zero-set analysis. Dense grids and optimizers independently support the result but do not establish it. No distinct exact neutral mutant survives. K=4 B has a unilateral tie manifold, resolved by a negative first-order coefficient.',
          'The conclusion uses the requested definition: for each fixed distinct pure mutant, mutant fitness is lower at sufficiently small positive frequency. The frequency bound may depend on the mutant. No full-SU(2), entanglement, noise, dynamics, mixed-mutant, or asymmetric-population claim is made.',
          '## Frozen Step 5 baseline',
          'Steps 1-4: 5827d9aaf631f67a02713eb8643609a7730707d2. Step 5 validated head: b174ec298187655e8d2eb143fe5509d7fcbc60d8. Step 5 PR #5 was squash merged as 655c86574b6c989f25153c7ad500b5b15d6e8316; the merged tree exactly matches the validated tree.',
          'All 58 existing tests passed, with zero failures or skips. Both original and Mac Mini saved validators passed: 5,487 and 5,491 detailed rows respectively; 16/16 global optimizer successes in each; two rejected / four unresolved classifications unchanged. Reconstructed coefficients and frequency diagnostics matched exactly. Fresh six-pair comparisons in each dataset differed by at most 3.56e-14. K=3 B and K=5 B remain rejected at orders 1 and 3 respectively.',
          'The primary checkout was switched to main by another process during work. Step 6 development continued in the isolated local worktree tmp/step6-worktree under the Mac project; no experiments or Git operations ran on WorkHub.',
          '## Question, model, and method',
          'For K=2 Q=(0,pi/2), K=4 Q=(0,pi/2), K=4 B=(pi/2,pi/2), and K=5 A=(0,2pi/5), decide the sign of the first genuinely nonzero coefficient of DeltaPi(epsilon)=sum_m a_m epsilon^m for every physically distinct mutant.',
          'Domain: 0<=theta<=pi and 0<=phi<=pi/2. Entanglement: gamma=pi/2. Strategy matrix: [[exp(i phi) cos(theta/2),sin(theta/2)],[-sin(theta/2),exp(-i phi) cos(theta/2)]]. The parity-adjusted generator is G=D^tensorK for even K and iD^tensorK for odd K; J=(I+iG)/sqrt(2). Payoffs sum pairwise PD (3,0,5,1). Population sampling is infinite and well mixed.',
          'For n=K-1, b_j=u_M(j)-u_R(j) and a_m=C(n,m) sum_{j<=m} (-1)^(m-j) C(m,j)b_j. The K=2 implementation gives a0=u(M,R)-u(R,R), a1=u(M,M)-u(R,M)-a0; on an exact a0 tie it is precisely the Maynard-Smith second condition.',
          'Use x=cos(theta/2)cos(phi), y=cos(theta/2)sin(phi), s=sin(theta/2), all nonnegative with x^2+y^2+s^2=1. Exact endpoint state amplitudes yield polynomial composition payoffs. SymPy verifies the certificate identities with zero polynomial remainders. A separate 80-digit dense tensor implementation checks selected points and derivatives.',
          '## Complete global sign certificate',table(['Resident','Exact a0','Zero-set resolution'],proof),
          'For Q and A, each non-self mutant has a0<0. For B, a0=-(9/2)s*x<=0. Its complete zero set is s=0 union x=0, corresponding to theta=0 and phi=pi/2, including canonical D. On these sets a1 is strictly negative except at B. Thus every admissible distinct mutant has a negative leading coefficient, and continuity of its finite polynomial gives a sufficiently small positive invasion-frequency interval. Full derivation and sign arguments are in docs/step6_methods.md; full exact polynomials are saved in symbolic_coefficients.json.',
          '## Local derivatives and feasible boundaries',
          table(['Resident','Feasible cone','Gradient a0','Hessian a0'],[
              ['K2 Q','dtheta>=0, dphi<=0','(0,0)','diag(-1.5,-4)'],['K4 Q','dtheta>=0, dphi<=0','(0,0)','diag(-4.5,-12)'],
              ['K4 B','dtheta free, dphi<=0','(0,2.25)','zero matrix'],['K5 A','dtheta>=0, dphi free','(0,0)','diag(-5.045084972,-16)']]),
          'Q and A have negative definite a0 Hessians on every nonzero feasible direction. B instead has a negative inward directional derivative because dphi<=0. Along the tangent dphi=0, a0 vanishes exactly and a1=-(3/4)sin(dtheta)^2=-(3/4)dtheta^2+(1/4)dtheta^4+O(dtheta^6). The zero a0 Hessian is therefore not evidence of neutrality. All coefficient gradients/Hessians and independent checks are saved, including eigendirections and quartic expansions.',
          'The local ball has angle radius 0.1. Outside it, the exact identities still cover the entire domain. For Q and A, a0 is bounded above by -min(alpha*sin(0.1/(2sqrt(2)))^2, beta*cos(0.1/(2sqrt(2)))^2*sin(0.1/sqrt(2))^2)<0. For B, a0<0 off its tie edges; on the remaining tie edges a1<=-(3/4)sin(0.1)^2 or a1=-3/4. No interval optimizer is needed to fill a gap in this coverage.',
          '## Grid convergence',
          'Canonicalization gives 221, 841, and 3281 operations, or 220, 840, and 3280 mutants after self exclusion for each resident. Every focal position and co-player placement was evaluated. There were no apparent neutral candidates and no positive leading coefficients in these three finite grids. The closest strongest grid candidates approach self as the grid is refined; this explains why maximum fitness differences approach zero.',
          table(['Resident','21x11 max DeltaPi(.01)','41x21','81x41'],[[n]+[f"{float(r['maximum_delta_001']):.9g}" for r in conv if r['resident']==n] for n in RESIDENTS]),
          '![Grid convergence](../../results/step6/plots/grid_convergence.png)',
          '## Per-resident inspection',
          'The entries below are the strongest non-self mutants on the finest grid under DeltaPi(0.01). There is no attained strongest non-self mutant in the continuous near-self limit. The optimizer near-self candidates are reported separately with high-precision corrections. Full polynomial arrays below are ordinary-precision values; tiny trailing entries are residuals, not exact identities.']
    locals_={
        'K2_Q':'Negative definite a0 Hessian; no null direction and no non-self unilateral tie.',
        'K4_Q':'Negative definite a0 Hessian; no null direction and no non-self unilateral tie.',
        'K4_B':'Linear rejection into the interior; negative quadratic a1 along the boundary tangent. The two complete unilateral tie edges contain no distinct neutral mutant.',
        'K5_A':'Negative definite a0 Hessian despite the theta boundary; unique phase zero at 2pi/5.'}
    for r in summary:
        name=r['resident'];k,rt,rp,payoff=RESIDENTS[name]
        coefficients=', '.join(f'{v:.12g}' for v in json.loads(r['power_coefficients']))
        pr=next(x for x in prec if x['resident']==name and x['probe']=='optimizer_best_double')
        text += [f'### {name}',table(['Quantity','Value'],[
            ['K; resident (theta,phi); payoff',f'{k}; ({rt:.12g}, {rp:.12g}); {payoff:g}'],
            ['Step 5 status','neutral_or_weak_candidate'],['Step 6 status','proven_ESS_restricted_EWL'],
            ['Strongest finest-grid mutant',f"({float(r['mutant_theta']):.12g}, {float(r['mutant_phi']):.12g})"],
            ['Distance from resident',f"{float(r['distance']):.12g} radians"],
            ['Leading order; coefficient',f"{r['leading_order']}; {float(r['leading_coefficient']):.12g}"],
            ['All power coefficients','['+coefficients+']'],['DeltaPi(0.01)',f"{float(r['delta_001']):.12g}"],
            ['Local result',locals_[name]],['Global result','Exact sign/zero-set proof; dense and seeded complement searches agree.'],
            ['Neutral set','No distinct exact neutral mutant.'],['Best double optimizer candidate at 80 digits',f"leading order {pr['high_precision_leading_order']}, coefficient {float(pr['high_precision_leading_value']):.8g} (negative)"]])]
    max_double=max(float(r['max_double_error']) for r in prec)
    local=json.loads((DATA/'local_derivative_analysis.json').read_text())
    max_der=max(x['maximum_error_to_symbolic_float'] for v in local.values() for x in v['independent_check']['coefficients'])
    failed=[r for r in opt if r['success']=='False']
    text += ['## Coefficient searches and precision',
        f"The saved search contains {len(opt)} runs: 52/52 differential-evolution global runs converged and 31/32 L-BFGS-B runs converged. The failed local run is {failed[0]['resident']} from start {failed[0]['start']}, with message '{failed[0]['message']}' and objective a0={float(failed[0]['objective_value']):.5g}; it is retained as failed. This numerical limit does not enter the analytic proof.",
        'Stage 0 maximizes a0 on the rectangle and on distance>=0.1, with three fixed seeds. All four boundaries are searched explicitly. Stage 1 for B uses exact edge parametrizations and complement subintervals. No loose threshold defines a tie constraint. Configurations, seeds, starts, iteration counts, evaluation counts, messages, final coordinates, and polynomials are retained.',
        f"There are {len(prec)} independent 80-digit checks, including displacements of 1e-20 and both B tie edges. Maximum ordinary/high-precision coefficient difference is {max_double:.3g}. Real centered 80-digit finite differences with step 1e-12 check every saved gradient/Hessian; maximum error after comparison in ordinary precision is {max_der:.3g}. The 1e-60 high-precision threshold is diagnostic only. Exact neutrality is decided algebraically.",
        table(['Audit','Maximum residual'],[[k,f'{v:.3g}'] for k,v in validation['maximum_errors'].items()]+[['Saved coefficient/fitness reconstruction',f"{validation['reconstruction_error']:.3g}"]]),
        '![B tie set](../../results/step6/plots/B_tie_set.png)',
        '![Near-self scaling](../../results/step6/plots/near_self_scaling.png)',
        '## Literature, novelty, and Step 7',
        'The project references A. Iqbal and A.H. Toor, Evolutionarily Stable Strategies in Quantum Games, Physics Letters A 280 (5-6), 249-256 (2001), arXiv:quant-ph/0007100v3, DOI 10.1016/S0375-9601(01)00082-2. Their two-player discussion concerns restricted strategy access and the ESS conditions. The K=2 result here is independently recovered, not a novel claim. See https://arxiv.org/abs/quant-ph/0007100v3.',
        'The related project reference is S.C. Benjamin and P.M. Hayden, Comment on Quantum Games and Quantum Strategies, Physical Review Letters 87, 069801 (2001), https://arxiv.org/abs/quant-ph/0003036. Admissible quantum strategy families can change stability conclusions. Restricted-family ESS does not establish full-SU(2) ESS; another game-specific result cannot be imported without matching the payoff, entangler, strategy set, and evolutionary definition.',
        'Recommended next action: research review of the exact K=4 B tie-set calculation and K=5 radical identity. After review, explicitly authorize Step 7: add the third unitary parameter, first perform the full-SU(2) unilateral Nash screen, and apply population ESS analysis only to its survivors.',
        'Future only: Step 8 re-identifies symmetric Nash candidates at each entanglement value; Step 9 adds explicit physical noise; Step 10 studies replicator dynamics. Polymorphic/asymmetric resident populations remain a later project. None was executed here.',
        '## Reproduction and limits',
        f"The numerical experiment commit is {meta['git_head_at_run']}. Runtime: {meta['runtime_seconds']:.3f} seconds for the recorded full experiment, including symbolic and precision work. This excludes implementation, baseline validation, report rendering, and tests. Hardware: {meta['hardware']}, {meta['cpu_count']} logical CPUs; workers: 1. Python: {meta['python'].splitlines()[0]}. Seeds: 20261002, 20261003, 20261004. The complete run settings and exact dependencies are in run_metadata.json and requirements-step6.txt.",
        'Commands, executed from the local Step 6 worktree:',
        '```sh\nPYTHONPATH=src studio-python -m unittest discover -s tests\nstudio-python scripts/validate_step5_results.py\nstudio-python scripts/validate_step5_results.py --results-dir results/step5_macmini_20260828\nPYTHONPATH=src studio-python scripts/audit_step5_for_step6.py\nPYTHONPATH=src studio-python scripts/run_step6.py --output-dir results/step6\nPYTHONPATH=src studio-python scripts/validate_step6_results.py\nPYTHONPATH=src studio-python scripts/build_step6_report.py\nPYTHONPATH=src studio-python scripts/build_step6_notebook.py\n```',
        'The full suite passes 75 tests, with zero failures and zero skips. The independent saved-data validator passes all 17,360 rows and six exact identity checks. The full test outcome is saved separately in results/step6/test_results.txt. Requirements use an isolated environment; the Mac Mini repeat is preserved as separate baseline evidence in the portable delivery.',
        'What is proved: pointwise rare-mutant stability of these four residents over the entire specified pure restricted-EWL domain. What is numerical: grid convergence values, optimizer outcomes, residual/error measurements, and finite-difference cross-checks. Remaining limitation: this is a reviewable analytic/symbolic certificate, not machine verification in a proof assistant. No unresolved domain or null-direction gap remains within the stated problem. No assertion about a uniform invasion barrier or enlarged strategy access is made.',
        '## Artifact index',
        table(['Path relative to project/worktree','Contents'],[
            ['results/step6/mutant_coefficients.csv.gz','All 17,360 mutants; full compositions, coefficients, aliases, focal arrays, diagnostics and errors'],
            ['results/step6/grid_convergence.csv; resident_summary.csv','Three-resolution convergence and four resident summaries'],
            ['results/step6/coefficient_optimizer_runs.csv','All 84 optimizer attempts, including the failed local refinement'],
            ['results/step6/precision_validation.csv','26 independent 80-digit probes; tiny values retained as strings'],
            ['results/step6/local_derivative_analysis.json','All coefficient derivatives, feasible cones, Taylor expansions, independent checks'],
            ['results/step6/symbolic_coefficients.json; certification_summary.json; neutral_sets.json','Exact formulas, zero-set identities and certificate'],
            ['results/step6/run_metadata.json; data_manifest.json; baseline_audit.json','Runtime provenance, hashes and frozen baseline validation'],
            ['results/step6/plots/','Four plots, each PNG and vector PDF'],
            ['docs/step6_methods.md','Detailed mathematical argument, methods and reproduction'],
            ['notebooks/step6_restricted_ewl_ess_resolution.ipynb','Executed inspection notebook'],
            ['output/step6/Step6_Restricted_EWL_ESS_Resolution.md and .pdf','This research report']]),
        '![a0 landscapes](../../results/step6/plots/a0_landscapes.png)']
    md='\n\n'.join(text)+'\n';source=OUT/'Step6_Restricted_EWL_ESS_Resolution.md';source.write_text(md)
    render_pdf(source,OUT/'Step6_Restricted_EWL_ESS_Resolution.pdf')
    print(source);print(OUT/'Step6_Restricted_EWL_ESS_Resolution.pdf')


def render_pdf(source,target):
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyR',fontName='Helvetica',fontSize=9,leading=12.5,spaceAfter=7))
    styles.add(ParagraphStyle(name='CellR',fontName='Helvetica',fontSize=7.7,leading=10))
    styles.add(ParagraphStyle(name='HeadR',fontName='Helvetica-Bold',fontSize=7.7,leading=10,textColor=colors.white))
    styles['Title'].fontSize=23;styles['Title'].leading=27;styles['Title'].alignment=0
    for key in ['Title','Heading1','Heading2']:styles[key].textColor=colors.HexColor('#163e5d')
    def fmt(t):
        t=escape(t);t=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',t);return t
    lines=source.read_text().splitlines();story=[];i=0;width=516
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]);i+=1
            code_style=ParagraphStyle('code',fontName='Courier',fontSize=6.6,leading=9,spaceAfter=8)
            story.append(Preformatted('\n'.join(code),code_style));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[x.strip() for x in lines[i].strip('|').split('|')]
                if not all(re.fullmatch('[-: ]+',c or '-') for c in cells):rows.append(cells)
                i+=1
            n=len(rows[0]);widths={2:[170,346],3:[70,220,226],4:[78,155,105,178]}[n]
            if rows[0][0]=='Resident' and '21x11' in rows[0][1]:widths=[84,144,144,144]
            tab=Table([[Paragraph(fmt(c),styles['HeadR' if j==0 else 'CellR']) for c in row] for j,row in enumerate(rows)],colWidths=widths,repeatRows=1,hAlign='LEFT')
            tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#163e5d')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#edf4f7'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
            if n==2 and rows[0][0]=='Quantity':
                heading=story.pop();story.append(KeepTogether([heading,tab,Spacer(1,8)]))
            else:story.extend([tab,Spacer(1,8)])
            continue
        if line.startswith('!['):
            path=re.search(r'\]\((.+)\)',line).group(1);image=Image(str(source.parent/path));height=width*image.imageHeight/image.imageWidth
            story.append(Image(str(source.parent/path),width=width,height=height));i+=1;continue
        level='Title' if line.startswith('# ') else 'Heading1' if line.startswith('## ') else 'Heading2' if line.startswith('### ') else 'BodyR'
        story.append(Paragraph(fmt(re.sub(r'^#{1,3} ','',line)),styles[level]));i+=1
    def footer(c,d):
        c.saveState();c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#52667a'));c.drawString(48,765,'Q-ESS | Step 6 restricted-EWL ESS resolution');c.drawString(48,25,'Research review | v1 | 2 October 2026');c.drawRightString(564,25,str(d.page));c.restoreState()
    SimpleDocTemplate(str(target),pagesize=(612,792),leftMargin=48,rightMargin=48,topMargin=46,bottomMargin=43,title='Step 6 Restricted EWL ESS Resolution',author='Q-ESS Research').build(story,onFirstPage=footer,onLaterPages=footer)

if __name__=='__main__':build()
