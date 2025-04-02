import numpy as np
import matplotlib.pyplot as plt
import os, re, h5py
from scipy.stats import binom
import seaborn as sns
cbcolors = sns.color_palette("colorblind").as_hex()

plt.rcParams["font.size"] = 8

coherences = [1, 5, 10, 20, 40]

## Organize all histograms into individual arrays per coherence, collected in a dictionary

top_dir = os.path.join("decision_making_results")
files = os.listdir(top_dir)

##

with h5py.File(os.path.join(top_dir, files[10]), "r") as f:
    keys = list(f.keys())
print(len(keys))

##

def get_hists_from_file(fname, model):
    hists = dict((c, []) for c in coherences)

    def get_first_last(l):
        return [l[0], l[-1]]

    with h5py.File(fname, "r") as f:
        keys = [key for key in list(f.keys()) if model in key]
        seed_coh = [list(map(int, get_first_last(re.findall("\d+", key)))) for key in keys if model in key]

        for i, (seed, c) in enumerate(seed_coh):
            hist1 = f[f"{model}_seed_{int(seed)}_coherence_{c}"]["selective_1"][()]
            hist2 = f[f"{model}_seed_{int(seed)}_coherence_{c}"]["selective_2"][()]
            if hist1.size + hist2.size > 1: 
                hists[c].append((hist1, hist2))

    return hists

def concat_dicts(list_of_dicts):
    newdict = {}
    for key in list_of_dicts[0].keys():
        vals = [np.array(d[key]) for d in list_of_dicts]
        newdict[key] = np.concatenate(vals)

    return newdict


exact_hists = concat_dicts([get_hists_from_file(os.path.join(top_dir, f), "exact") for f in files])
approx_hists = concat_dicts([get_hists_from_file(os.path.join(top_dir, f), "approx") for f in files])


def get_correct_rate(hists):
    preds = 1 - hists.sum(-1).argmax(-1)
    return preds.mean()


exact_correct_preds = []
approx_correct_preds = []
for c in coherences:
    exact_correct_preds.append(get_correct_rate(exact_hists[c]))
    approx_correct_preds.append(get_correct_rate(approx_hists[c]))


## 


# bootstrapping
exact_correct = []
approx_correct = []
for i in range(5000):
    print(i, end="\r")
    exact_sample_correct = []
    approx_sample_correct = []
    for c in coherences:
        num = exact_hists[c].shape[0]
        sample_inds = np.random.choice(np.arange(num), size=num, replace=True)
        exact_sample = exact_hists[c][sample_inds]
        exact_sample_correct.append((1 - exact_sample.sum(-1).argmax(-1)).mean())

        num = approx_hists[c].shape[0]
        sample_inds = np.random.choice(np.arange(num), size=num, replace=True)
        approx_sample = approx_hists[c][sample_inds]
        approx_sample_correct.append((1 - approx_sample.sum(-1).argmax(-1)).mean())

    exact_correct.append(exact_sample_correct)
    approx_correct.append(approx_sample_correct)

exact_correct = np.array(exact_correct)
approx_correct = np.array(approx_correct)

exact_prob = exact_correct.mean(0)
approx_prob = approx_correct.mean(0)

# find 10th and 90th percentiles
exact_sortargs = np.argsort(exact_correct, axis=0)
lower_ind = int(len(exact_sortargs) * 0.05)
upper_ind = int(len(exact_sortargs) * 0.95)
exact_10th_percentile = exact_correct[exact_sortargs[lower_ind], np.arange(5)]
exact_90th_percentile = exact_correct[exact_sortargs[upper_ind], np.arange(5)]

approx_sortargs = np.argsort(approx_correct, axis=0)
lower_ind = int(len(exact_sortargs) * 0.05)
upper_ind = int(len(exact_sortargs) * 0.95)
approx_10th_percentile = approx_correct[approx_sortargs[lower_ind], np.arange(5)]
approx_90th_percentile = approx_correct[approx_sortargs[upper_ind], np.arange(5)]

## 

coherences = np.array(coherences)
fig, ax = plt.subplots(1)
fig.set_size_inches([7.6 / 2.54, 3.6 / 2.54])
fig.subplots_adjust(left=0.15, bottom=0.25, top=0.98, right=0.96)
x = np.linspace(0, 100, 1001)
ax.plot(x, 1 - 0.5 * np.exp(-(x / 9.2) ** 1.5), c="black", label="Weibull fit (Wang 2002)", zorder=0) 
ax.scatter(coherences * np.exp(-0.05), exact_correct.mean(0), label="Exact model", c=cbcolors[1], s=10, zorder=1)
ax.vlines(coherences * np.exp(-0.05), exact_10th_percentile, exact_90th_percentile, color=cbcolors[1], linestyle="solid", lw=1.5, zorder=1)
ax.scatter(coherences * np.exp(0.05), approx_correct.mean(0), label="Approximate model", c=cbcolors[0], s=10, zorder=1)
# shift slightly to make both confidence intervals visible
ax.vlines(coherences * np.exp(0.05), approx_10th_percentile, approx_90th_percentile, color=cbcolors[0], linestyle="solid", lw=1.5, zorder=1)
ax.set_xlabel("Coherence", fontsize=8, labelpad=0)
ax.set_ylabel("Accuracy", fontsize=8, labelpad=0)
yticks = ax.get_yticks()
yticklabels = [str(int(float(l.get_text())*100)) for l in ax.get_yticklabels()]
ax.set_yticklabels(yticklabels)
ax.set_ylim(0.5, 1.01)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.])
ax.set_yticklabels(["50", "60", "70", "80", "90", "100"])

ax.semilogx()
ax.set_xlim(0.92, 100)
plt.savefig("figure4.pdf")
plt.show()

## 

