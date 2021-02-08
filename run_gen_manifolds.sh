python3 gen_manifold.py --samples=3000 --model=StyleGAN2 --class=ffhq --c1=0 --c2=1 --use_w --layer=style -b=10_000
python3 gen_manifold.py --samples=3000 --model=StyleGAN2 --class=cat --c1=6 --c2=9 --use_w --layer=style -b=10_000
python3 gen_manifold.py --samples=3000 --model=StyleGAN2 --class=cat --c1=4 --c2=13 --use_w --layer=style -b=10_000
python3 gen_manifold.py --samples=3000 --model=StyleGAN2 --class=car --c1=0 --c2=13 --use_w --layer=style -b=10_000
python3 gen_manifold.py --samples=3000 --model=StyleGAN2 --class=kitchen --c1=3 --c2=8 --use_w --layer=style -b=10_000
python3 gen_manifold.py --samples=3000 --model=StyleGAN2 --class=bedrooms --c1=5 --c2=9 --sigma=1.5 --use_w --layer=style -b=10_000
python3 gen_manifold.py --samples=3000 --model=StyleGAN2 --class=church --c1=3 --c2=9 --use_w --layer=style -b=10_000
python3 gen_manifold.py --samples=3000 --model=StyleGAN2 --class=horse --c1=1 --c2=11 --use_w --layer=style -b=10_000

python3 gen_manifold.py --samples=6000 --model=StyleGAN2 --class=ffhq --end_c=8 --use_w --layer=style -b=10_000
python3 gen_manifold.py --samples=6000 --model=StyleGAN2 --class=cat --end_c=8 --use_w --layer=style -b=10_000
python3 gen_manifold.py --samples=6000 --model=StyleGAN2 --class=car --end_c=8 --use_w --layer=style -b=10_000
python3 gen_manifold.py --samples=6000 --model=StyleGAN2 --class=kitchen --end_c=8 --use_w --layer=style -b=10_000
python3 gen_manifold.py --samples=6000 --model=StyleGAN2 --class=bedrooms --end_c=8 -sigma=1.5  --use_w --layer=style -b=10_000
python3 gen_manifold.py --samples=6000 --model=StyleGAN2 --class=church --end_c=8 --use_w --layer=style -b=10_000
python3 gen_manifold.py --samples=6000 --model=StyleGAN2 --class=horse --end_c=8 --use_w --layer=style -b=10_000
