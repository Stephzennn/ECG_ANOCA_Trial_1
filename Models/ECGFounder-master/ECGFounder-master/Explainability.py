
import os
import pandas as pd
from matplotlib import pyplot as plt
import os
from net1d import Net1D
import torch
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import Lime
from captum.attr import NeuronConductance
from embedding import createDataloader, loadWeightsToModel, generateOutput
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import uniform_filter1d
if __name__ == "__main__":
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
    print(device)

    saved_dir = './res/eval'
    csv_filepath = './csv/ptbxl_label.csv'
    ecg_filepath = './data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    all_embeddings = []
    all_labels = []
    testset, testloader = createDataloader('./tasks.txt', 512, ecg_filepath,csv_filepath)
    model = Net1D(
        in_channels=12, 
        base_filters=64, #32 64
        ratio=1, 
        filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
        m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
        kernel_size=16, 
        stride=2, 
        groups_width=16,
        verbose= False, 
        use_bn=False,
        use_do=False,
        n_classes=150,
        return_features=True)

    model.to(device)


    log = loadWeightsToModel('./checkpoint/12_lead_ECGFounder.pth', model, device)

    #all_gt, all_embeddings, df_gt, labels, all_pred_prob, all_logits = generateOutput(testloader, model, device)

    # Testset
    input_x, input_y = next(iter(testloader))

    target = 5


    #----
    #CuDA out of memory
    test_input_tensor = input_x.to(device).float()
    test_input_tensor.requires_grad_()

    attr, delta = ig.attribute(
        test_input_tensor,
        target=5,
        return_convergence_delta=True
    )

    attr = attr.detach().cpu().numpy()
    delta = delta.detach().cpu().numpy()

    print("Attribution shape:", attr.shape)
    print("Delta shape:", delta.shape)

    #---


    #===

    import torch
    from captum.attr import IntegratedGradients

    target_index = 5  # Fifth label

    positive_indices = torch.where(
        input_y[:, target_index] > 0
    )[0]

    print("Target index:", target_index)
    #print("Target name:", labels[target_index])
    print("Matching ECG indices:", positive_indices)
    print("Number of matches:", len(positive_indices))
    print("Matching ECG indices:", positive_indices)
    print("Number of matches:", len(positive_indices))
    model.eval()


    # Return logits only because  model also returns deep features
    def forward_logits(x):
        logits, _ = model(x)
        return logits

    ig = IntegratedGradients(forward_logits)

    # Select only the first ECG—not the entire 512-record batch
    selected_index = positive_indices[1].item()

    test_input_tensor = (
        input_x[selected_index:selected_index + 1]
        .detach()
        .clone()
        .float()
        .to(device)
        .requires_grad_(True)
    )

    print("Selected ECG index:", selected_index)
    print("Target-10 ground truth:", input_y[selected_index, target_index].item())
    print("Input shape:", test_input_tensor.shape)

    torch.cuda.empty_cache()

    attr, delta = ig.attribute(
        test_input_tensor,
        target=target_index,
        n_steps=20,
        internal_batch_size=1,
        return_convergence_delta=True
    )

    attr = attr.detach().cpu().numpy()
    delta = delta.detach().cpu().numpy()

    print("Attribution shape:", attr.shape)
    print("Delta:", delta)


    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.ndimage import uniform_filter1d
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable


    ecg = test_input_tensor[0].detach().cpu().numpy()
    importance = np.maximum(attr[0], 0)  # (12, 5000)

    importance = uniform_filter1d(
        importance,
        size=10,
        axis=-1
    )

    # Calculate thresholds independently for each lead
    # Shapes are (12, 1), allowing broadcasting across time
    low = np.percentile(
        importance,
        70,
        axis=1,
        keepdims=True
    )

    high = np.percentile(
        importance,
        99,
        axis=1,
        keepdims=True
    )

    # Normalize each lead independently
    importance_scaled = np.clip(
        (importance - low) / (high - low + 1e-12),
        0,
        1
    )

    lead_names = [
        "I", "II", "III",
        "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]

    samples = np.arange(ecg.shape[1])

    fig, axes = plt.subplots(
        12,
        1,
        figsize=(22, 20),
        sharex=True,
        constrained_layout=True
    )

    for lead_index, ax in enumerate(axes):
        signal = ecg[lead_index]

        # Original blue ECG
        ax.plot(
            samples,
            signal,
            color="#1f77b4",
            linewidth=1.4,
            zorder=1
        )

        points = np.column_stack(
            [samples, signal]
        ).reshape(-1, 1, 2)

        segments = np.concatenate(
            [points[:-1], points[1:]],
            axis=1
        )

        segment_importance = (
            importance_scaled[lead_index, :-1]
            + importance_scaled[lead_index, 1:]
        ) / 2

        # RGBA: red with attribution-dependent opacity
        red_colors = np.zeros((len(segments), 4))
        red_colors[:, 0] = 1.0
        red_colors[:, 3] = segment_importance

        red_overlay = LineCollection(
            segments,
            colors=red_colors,
            linewidth=3.0,
            zorder=2
        )

        ax.add_collection(red_overlay)

        ax.set_xlim(0, ecg.shape[1] - 1)

        margin = max(np.ptp(signal) * 0.10, 1e-6)
        ax.set_ylim(
            signal.min() - margin,
            signal.max() + margin
        )

        ax.set_ylabel(lead_names[lead_index])
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time/sample")

    cmap = LinearSegmentedColormap.from_list(
        "blue_red",
        ["#1f77b4", "#ff0000"]
    )

    colorbar_object = ScalarMappable(
        norm=Normalize(0, 1),
        cmap=cmap
    )
    colorbar_object.set_array([])

    fig.colorbar(
        colorbar_object,
        ax=axes,
        label="Relative attribution importance",
        shrink=0.75,
        pad=0.01
    )

    fig.suptitle(
        f"Target {target_index} Attribution — ECG at Batch Index {selected_index}",
        fontsize=18
    )

    plt.show()
    #====

    # Global normalization 
    selected_index = positive_indices[2].item()
    
    test_input_tensor = (
        input_x[selected_index:selected_index + 1]
        .detach()
        .clone()
        .float()
        .to(device)
        .requires_grad_(True)
    )

    print("Selected ECG index:", selected_index)
    print("Target-10 ground truth:", input_y[selected_index, target_index].item())
    print("Input shape:", test_input_tensor.shape)

    torch.cuda.empty_cache()

    attr, delta = ig.attribute(
        test_input_tensor,
        target=target_index,
        n_steps=20,
        internal_batch_size=1,
        return_convergence_delta=True
    )

    attr = attr.detach().cpu().numpy()
    delta = delta.detach().cpu().numpy()

    print("Attribution shape:", attr.shape)
    print("Delta:", delta)


    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.ndimage import uniform_filter1d
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    import numpy as np


    def millisecond_formatter(x, position):
        """Label only whole-second positions in milliseconds."""
        if np.isclose(x, round(x), atol=1e-8):
            return f"{int(round(x * 1000))}"
        return ""



    ecg = test_input_tensor[0].detach().cpu().numpy()
    #importance = np.abs(attr[0])
    importance = np.maximum(attr[0], 0)  # (12, 5000)

    importance = uniform_filter1d(
        importance,
        size=10,
        axis=-1
    )

    # Calculate thresholds independently for each lead
    
    # Global thresholds across all leads and time samples
    low = np.percentile(importance, 70)
    high = np.percentile(importance, 99)

    # Normalize globally
    importance_scaled = np.clip(
        (importance - low) / (high - low + 1e-12),
        0,
        1
    )

    lead_names = [
        "I", "II", "III",
        "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6"
    ]

    samples = np.arange(ecg.shape[1])

    fig, axes = plt.subplots(
        12,
        1,
        figsize=(22, 20),
        sharex=True,
        constrained_layout=True
    )

    for lead_index, ax in enumerate(axes):
        signal = ecg[lead_index]
        # Keep grid behind the ECG waveform
        ax.set_axisbelow(True)

        # Standard ECG grid at 25 mm/s
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))   # 40 ms
        ax.xaxis.set_major_locator(MultipleLocator(0.20))   # 200 ms

        # Standard amplitude grid, assuming ECG is in mV
        ax.yaxis.set_minor_locator(MultipleLocator(0.10))   # 0.1 mV
        ax.yaxis.set_major_locator(MultipleLocator(0.50))   # 0.5 mV

        # ECG-paper-style grid
        ax.grid(
            which="minor",
            color="#f8cccc",
            linewidth=0.45,
            alpha=0.80
        )

        ax.grid(
            which="major",
            color="#e99a9a",
            linewidth=0.90,
            alpha=0.90
        )

        # Strong vertical marker every 1 second
        start_second = int(np.floor(samples[0]))
        end_second = int(np.ceil(samples[-1]))

        for second in range(start_second, end_second + 1):
            ax.axvline(
                second,
                color="#cf6f6f",
                linewidth=1.15,
                alpha=0.85,
                zorder=0
            )

        # Original blue ECG
        ax.plot(
            samples,
            signal,
            color="#1f77b4",
            linewidth=1.4,
            zorder=1
        )

        points = np.column_stack(
            [samples, signal]
        ).reshape(-1, 1, 2)

        segments = np.concatenate(
            [points[:-1], points[1:]],
            axis=1
        )

        segment_importance = (
            importance_scaled[lead_index, :-1]
            + importance_scaled[lead_index, 1:]
        ) / 2

        # RGBA: red with attribution-dependent opacity
        red_colors = np.zeros((len(segments), 4))
        red_colors[:, 0] = 1.0
        red_colors[:, 3] = segment_importance

        red_overlay = LineCollection(
            segments,
            colors=red_colors,
            linewidth=3.0,
            zorder=2
        )

        ax.add_collection(red_overlay)

        ax.set_xlim(0, ecg.shape[1] - 1)

        margin = max(np.ptp(signal) * 0.10, 1e-6)
        ax.set_ylim(
            signal.min() - margin,
            signal.max() + margin
        )

        ax.set_ylabel(lead_names[lead_index])
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time/sample")

    cmap = LinearSegmentedColormap.from_list(
        "blue_red",
        ["#1f77b4", "#ff0000"]
    )

    colorbar_object = ScalarMappable(
        norm=Normalize(0, 1),
        cmap=cmap
    )
    colorbar_object.set_array([])

    fig.colorbar(
        colorbar_object,
        ax=axes,
        label="Relative attribution importance",
        shrink=0.75,
        pad=0.01
    )

    fig.suptitle(
        f"Target {target_index} Attribution — ECG at Batch Index {selected_index}",
        fontsize=18
    )

    plt.show()
#=============================

#Atrribution function

def plot_attribution(test_input_tensor, attr, target_index, selected_index, fs=500):
    to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    ecg = to_np(test_input_tensor[0])                 # (12, n)
    importance = np.maximum(to_np(attr[0]), 0)        # (12, n)
    """
    
    importance = uniform_filter1d(importance, size=10, axis=-1)

    low, high = np.percentile(importance, [70, 99])
    importance_scaled = np.clip((importance - low) / (high - low + 1e-12), 0, 1)

    """
    low  = np.zeros((importance.shape[0], 1))
    high = np.ones((importance.shape[0], 1))
    for i, row in enumerate(importance):
        nz = row[row > 0]
        if nz.size:
            low[i, 0], high[i, 0] = np.percentile(nz, [70, 99])

    importance_scaled = np.clip((importance - low) / (high - low + 1e-12), 0, 1)
    #"""
    
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]

    n_samples = ecg.shape[1]
    time = np.arange(n_samples) / fs                  
    duration = (n_samples - 1) / fs

    fig, axes = plt.subplots(12, 1, figsize=(30, 20),
                             sharex=True, constrained_layout=True)

    for lead_index, ax in enumerate(axes):
        signal = ecg[lead_index]
        ax.set_axisbelow(True)

        # 25 mm/s: minor = 40 ms, major = 200 ms
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))
        ax.xaxis.set_major_locator(MultipleLocator(0.20))
        # 10 mm/mV: minor = 0.1 mV, major = 0.5 mV
        ax.yaxis.set_minor_locator(MultipleLocator(0.10))
        ax.yaxis.set_major_locator(MultipleLocator(0.50))

        ax.grid(which="minor", color="#f8cccc", linewidth=0.45, alpha=0.80)
        ax.grid(which="major", color="#e99a9a", linewidth=0.90, alpha=0.90)

        for second in range(int(np.floor(duration)) + 1):
            ax.axvline(second, color="#cf6f6f", linewidth=1.15,
                       alpha=0.85, zorder=0)

        ax.plot(time, signal, color="#1f77b4", linewidth=1.4, zorder=1)

        points = np.column_stack([time, signal]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        segment_importance = (importance_scaled[lead_index, :-1]
                              + importance_scaled[lead_index, 1:]) / 2

        red_colors = np.zeros((len(segments), 4))
        red_colors[:, 0] = 1.0
        red_colors[:, 3] = segment_importance

        ax.add_collection(LineCollection(segments, colors=red_colors,
                                         linewidth=3.0, zorder=2))

        ax.set_xlim(0, duration)
        margin = max(np.ptp(signal) * 0.10, 1e-6)
        ax.set_ylim(signal.min() - margin, signal.max() + margin)
        ax.set_ylabel(lead_names[lead_index])
        # note: no ax.grid(alpha=0.2) here — it would clobber the styling above

    axes[-1].set_xlabel("Time (s)")

    cmap = LinearSegmentedColormap.from_list("blue_red", ["#1f77b4", "#ff0000"])
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(),
                 label="Relative attribution importance", shrink=0.75, pad=0.01)

    fig.suptitle(f"Target {target_index} Attribution — ECG at Batch Index {selected_index}",
                 fontsize=18)
    plt.show()


target_index = 5  # Fifth label

positive_indices = torch.where(
    input_y[:, target_index] > 0
)[0]

print("Target index:", target_index)
#print("Target name:", labels[target_index])
print("Matching ECG indices:", positive_indices)
print("Number of matches:", len(positive_indices))
print("Matching ECG indices:", positive_indices)
print("Number of matches:", len(positive_indices))
model.eval()


# Return logits only because  model also returns deep features
def forward_logits(x):
    logits, _ = model(x)
    return logits

ig = IntegratedGradients(forward_logits)

# Select only the first ECG—not the entire 512-record batch
selected_index = positive_indices[2].item()

test_input_tensor = (
    input_x[selected_index:selected_index + 1]
    .detach()
    .clone()
    .float()
    .to(device)
    .requires_grad_(True)
)

print("Selected ECG index:", selected_index)
print("Target-10 ground truth:", input_y[selected_index, target_index].item())
print("Input shape:", test_input_tensor.shape)
selected_index = positive_indices[1].item()
    
test_input_tensor = (
    input_x[selected_index:selected_index + 1]
    .detach()
    .clone()
    .float()
    .to(device)
    .requires_grad_(True)
)

print("Selected ECG index:", selected_index)
print("Target-10 ground truth:", input_y[selected_index, target_index].item())
print("Input shape:", test_input_tensor.shape)

torch.cuda.empty_cache()

attr, delta = ig.attribute(
    test_input_tensor,
    target=target_index,
    n_steps=20,
    internal_batch_size=1,
    return_convergence_delta=True
)

attr = attr.detach().cpu().numpy()
delta = delta.detach().cpu().numpy()
plot_attribution(test_input_tensor, attr, target_index, selected_index, fs=500)



#============
    type(testset.data)

    print(testset.data.shape[0])
    # The testset takes in the number of samples.
    dd = testset.__getitem__(21798)

    print(dd[0].shape)
    type(dd)

    inputTrial = []

    #print()
    for x in range(testset.data.shape[0]):
            if x % 1000 == 0 :
                print(x)
            inputTrial.append(testset.__getitem__(x)[0] )

    len(inputTrial)
    print(inputTrial.shape)