"""Plot the final Gaussian power settings with conditional Wilson intervals."""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACKAGE = Path(__file__).resolve().parent
data = json.loads((PACKAGE / "outputs/final_tables/final_results.json").read_text())["simulation"]["settings"]
budgets = (8192,32768,131072,524288)
colors = ("#0072B2","#D55E00","#009E73")
fig,axes = plt.subplots(2,2,figsize=(8.0,6.2),sharex=True,sharey=True)
for panel,(template,axis) in enumerate(zip((0,1,3,4),axes.flat)):
    for gap,color in zip(("1/8","1/4","1/2"),colors):
        selected = [next(row for row in data if row["template"]==template and row["gap"]==gap and row["rows"]==n)
                    for n in budgets]
        count = np.array([row["reports"] for row in selected],dtype=float)
        size = np.array([row["cases"] for row in selected],dtype=float)
        probability = count/size
        z = 1.959963984540054
        denominator = 1+z*z/size
        center = (probability+z*z/(2*size))/denominator
        radius = z*np.sqrt(probability*(1-probability)/size+z*z/(4*size*size))/denominator
        axis.fill_between(budgets,center-radius,center+radius,color=color,alpha=.12,lw=0)
        axis.plot(budgets,probability,"o-",color=color,lw=1.5,markersize=3.8,label="Gap "+gap)
    axis.axhline(0.,color="#595959",ls="--",lw=1,label="Three region references")
    axis.set_xscale("log",base=2)
    axis.set_xticks(budgets,[f"{n:,}" for n in budgets],fontsize=8)
    axis.set_yticks(np.linspace(0,1,6))
    axis.tick_params(axis="y",labelsize=8)
    axis.set_ylim(-.045,1.04)
    axis.grid(axis="y",alpha=.25,lw=.6)
    axis.set_axisbelow(True)
    axis.spines[["right","top"]].set_visible(False)
    axis.set_title(chr(65+panel)+"   Template "+str(template),loc="left",fontsize=10)
    if panel%2==0:
        axis.set_ylabel("Shortcut detection probability",fontsize=9)
    if panel>=2:
        axis.set_xlabel("Rows per experimental cell",fontsize=9)
handles,labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(.5,-.015))
fig.tight_layout(rect=(0,.07,1,1))
output = PACKAGE / "outputs/final_tables"
for suffix in ("png","pdf","svg"):
    fig.savefig(output / ("independent_power376_v1."+suffix),dpi=300,bbox_inches="tight")
plt.close(fig)
print("FINAL_POWER_FIGURE_GENERATED")
