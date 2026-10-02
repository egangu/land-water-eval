"""Recreate the four panels from ground truth and lossless prediction maps."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from common import DATA, load_grid, map_image

def plot(args):
    _, truth = load_grid(args.csv)
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((results/'summary.json').read_text())
    panels = [('Ground truth', np.asarray(map_image(truth)), None)]
    names = [('base','Gemma 4 31B Base'),('direct','Gemma 4 31B IT w/o reasoning'),('reasoning','Gemma 4 31B IT w/ reasoning')]
    for mode, name in names:
        with Image.open(results/(mode+'.png')) as image:
            pixels = np.asarray(image.convert('RGBA'))
        assert pixels.shape == (90,180,4)
        panels.append((name,pixels,summary[mode]['accuracy']))
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(2,2,figsize=(13,7.6),facecolor='white')
    fig.subplots_adjust(left=.035,right=.965,bottom=.078,top=.915,wspace=.075,hspace=.26)
    for ax, (name,pixels,accuracy) in zip(axes.flat,panels,strict=True):
        missing = pixels[:,:,3] == 0
        display = pixels[:,:,0].copy()
        display[missing] = 255
        ax.imshow(display,cmap='gray',vmin=0,vmax=255,interpolation='nearest',origin='upper')
        for row,col in np.argwhere(missing):
            ax.plot([col-.38,col+.38],[row-.38,row+.38],color='black',lw=.4,solid_capstyle='butt')
            ax.plot([col-.38,col+.38],[row+.38,row-.38],color='black',lw=.4,solid_capstyle='butt')
        ax.set(xlim=(-.5,179.5),ylim=(89.5,-.5),xticks=[],yticks=[])
        for spine in ax.spines.values(): spine.set_linewidth(.65)
        title = name if accuracy is None else f'{name} · {accuracy*100:.2f}%'
        ax.text(0,1.065,title,transform=ax.transAxes,ha='left',va='bottom',fontsize=14,fontweight='semibold')
    fig.text(.5,.034,'2° grid  ·  White: land  ·  Black: water  ·  ×: no answer',ha='center',fontsize=10)
    for extension in ('png','pdf','svg'): fig.savefig(out/f'land_water_comparison.{extension}',dpi=240)
    plt.close(fig)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',default=str(DATA))
    parser.add_argument('--results',default='results')
    parser.add_argument('--out',default='figures')
    plot(parser.parse_args())
