from pathlib import Path
import subprocess, tempfile, os
import argparse

parser = argparse.ArgumentParser(description="Check centralized case reading and optional VTK equivalence.")
parser.add_argument("--reference", type=Path, help="Executable built before the parameter refactor")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
executables = [("after", root / "solver")]
if args.reference:
    executables.insert(0, ("before", args.reference.resolve()))
env=dict(os.environ, OMPI_MCA_btl='self')
base='''nx=12
ny=8
ng=3
finalTime=0.001
cfl=0.05
outputEvery=0
outPrefix=check
output.vtkDataOutputType=cell
setFields.bg.rho=1
setFields.bg.p=1
setFields.bg.u=0.1
setFields.bg.v=0.2
'''
variants={
'defaults':'',
'weno_regions':'''reconstruction.scheme=weno5
setFields.nRegions=2
setFields.region1.xMax=0.5
setFields.region1.rho=0.9
setFields.region2.yMin=0.1
setFields.region2.u=0.15
''',
'shock':'''setFields.nRegions=1
setFields.region1.shockMach=1.2
setFields.region1.shockDir=-y
setFields.region1.yMax=0.1
reconstruction.scheme=muscl
reconstruction.limiter=minmod
''',
'boundaries':'''bc=slipWall
bc.left=inlet
bc.inlet.rho=1
bc.inlet.u=0.1
bc.inlet.v=0.2
bc.inlet.p=1
bc.right=outlet
bc.outlet.p=1
bc.top=symmetry
''',
'side_inlet':'''bc.left=inlet
bc.inlet.rho=9
bc.left.inlet.rho=1
bc.left.inlet.u=0.1
bc.left.inlet.v=0.2
bc.left.inlet.p=1
output.vtkDataOutputType=PoInT
'''}
with tempfile.TemporaryDirectory(prefix='euler_parameters_') as tmp:
 for name, extra in variants.items():
  outputs=[]
  for label,exe in executables:
   work=Path(tmp)/name/label; work.mkdir(parents=True)
   case=work/'case.cfg';case.write_text(base+extra)
   p=subprocess.run([str(exe),str(case)],cwd=work,env=env,capture_output=True,text=True)
   assert p.returncode==0,(name,label,p.stdout,p.stderr)
   outputs.append(sorted((p.name,p.read_bytes()) for p in (work/'solution').glob('*.vtk')))
  assert outputs[0], name
  if args.reference:
   assert outputs[0] == outputs[1], name
  print(name + (': identical VTK output' if args.reference else ': passed'))
 for name,extra in {'integer':'nx=12oops','finite':'cfl=nan','bool':'writeFinal=maybe','inlet':'bc.left=inlet','limiter':'reconstruction.limiter=bad','shock':'setFields.nRegions=1\nsetFields.region1.shockMach=1'}.items():
  work=Path(tmp)/('invalid_'+name);work.mkdir();case=work/'case.cfg';case.write_text(base+extra+'\n')
  p=subprocess.run([str(root/'solver'),str(case)],cwd=work,env=env,capture_output=True,text=True)
  assert p.returncode!=0,(name,p.stdout,p.stderr)
  print('invalid '+name+': rejected')
