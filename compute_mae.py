import os
import glob
import pickle
import numpy as np
from scipy.ndimage import label

def extract_events(contact_array):
    """Extract independent stepping events"""
    labeled_array, num_features = label(contact_array)
    events = []
    for i in range(1, num_features + 1):
        indices = np.where(labeled_array == i)[0]
        events.append((indices[0], indices[-1], len(indices)))
    return events

def get_all_errors_ms(gt_array, pred_array, fps=50):
    """Calculate the list of MAE errors for a single foot"""
    ms_per_frame = 1000 / fps
    gt_events = extract_events(gt_array)
    pred_events = extract_events(pred_array)
    
    if not gt_events: return []
    
    errors_ms = []
    for gt_start, gt_end, gt_dur in gt_events:
        gt_mid = (gt_start + gt_end) / 2
        if pred_events:
            # Find the closest predicted step in time
            closest_pred = min(pred_events, key=lambda p: abs((p[0]+p[1])/2 - gt_mid))
            # Calculate the error in duration for this step
            errors_ms.append(abs(gt_dur - closest_pred[2]) * ms_per_frame)
        else:
            errors_ms.append(gt_dur * ms_per_frame) # Missed detections count as full errors
            
    return errors_ms

if __name__ == "__main__":
    # Ensure this path is correct
    pkl_files = glob.glob("Results/BODY25/footformer/com_contact_pressure/kld_def/seq_9_lr_0.0002_bs_512__adamw__cosine_warmup_pose_dim512_heads_8_layers_4_dropout_0.2_pos_learnable_decoder_dim_128_mlpd_1024_embedder_gcn_multi/eval/output/subject*_output.pkl")
    
    if not pkl_files:
        print("No pkl files found, please check the path!")
        exit()
        
    all_subject_maes = []
    
    # Lists to store text and data
    summary_lines = []
    csv_lines = ["Subject,Foot,Error_ms\n"] # CSV Header
    
    for pkl_path in pkl_files:
        subj_name = os.path.basename(pkl_path).split('_')[0]
        print(f"\n--- Processing: {subj_name} ---")
        summary_lines.append(f"--- Processing: {subj_name} ---")
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        # this is a dict with keys: 'targets' and 'predictions', each containing 'contact' array of shape (N, 2)   
        gt_contact = data['targets']['contact']       
        pred_contact = data['predictions']['contact'] 
        
        # Binarize
        gt_binary = (gt_contact > 0.5).astype(int)
        pred_binary = (pred_contact > 0.5).astype(int)
        
        # Get all raw error data
        errors_left = get_all_errors_ms(gt_binary[:, 0], pred_binary[:, 0])
        errors_right = get_all_errors_ms(gt_binary[:, 1], pred_binary[:, 1])
        
        # Write to CSV
        for err in errors_left:
            csv_lines.append(f"{subj_name},Left,{err:.2f}\n")
        for err in errors_right:
            csv_lines.append(f"{subj_name},Right,{err:.2f}\n")
            
        # Calculate means
        mae_left = np.mean(errors_left) if errors_left else np.nan
        mae_right = np.mean(errors_right) if errors_right else np.nan
        subject_mae = np.nanmean([mae_left, mae_right])
        
        all_subject_maes.append(subject_mae)
        
        # Print and collect summary
        res_text = (f"Left foot MAE: {mae_left:.2f} ms\n"
                    f"Right foot MAE: {mae_right:.2f} ms\n"
                    f"Average MAE: {subject_mae:.2f} ms\n")
        print(res_text, end="")
        summary_lines.append(res_text)
        
    # Calculate final average and print
    final_avg = np.nanmean(all_subject_maes)
    final_text = f"\n================ Final Results ================\nOverall Subject Contact Time MAE: {final_avg:.2f} ms\n"
    print(final_text)
    summary_lines.append(final_text)
    
    # ---------------- Core saving module ----------------
    # 1. Save summary report file
    with open("summary_average_mae.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("✅ Summary report saved to: summary_average_mae.txt")
        
    # 2. Save all raw error details CSV
    with open("all_contact_errors.csv", "w") as f:
        f.writelines(csv_lines)
    print("✅ All error details saved to: all_contact_errors.csv")