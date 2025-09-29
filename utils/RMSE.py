# --- RMSE 계산 코드 시작 ---
import math

def compute_rmse_between_totals_and_real_totals(totals, real_totals):
    all_keys = set(totals.keys()) & set(real_totals.keys())
    total_error_ratios = []

    squared_errors = []
    print(f"Calculating RMSE between estimated and real traffic keys: {len(all_keys)}")
    for key in all_keys:
        # real_totals 값 (실제값)
        true_val = real_totals.get(key, 0)
        # totals 값 (추정값)
        pred_val = totals.get(key, 0)
        error = true_val - pred_val
        error_ratio = abs(error)/max(true_val, pred_val)

        total_error_ratios.append(error_ratio)
        # print(f"Key: {key}, True: {true_val}, Pred: {pred_val}, Error: {error}, Error Ratio: {error_ratio}")
        squared_error = error ** 2
        squared_errors.append(squared_error)

    # 3. RMSE 계산
    if squared_errors:
        mean_squared_error = sum(squared_errors) / len(squared_errors)
        mean_error_ratio = sum(total_error_ratios) / len(total_error_ratios)
        rmse = math.sqrt(mean_squared_error)
        print(f"RMSE between estimated and real traffic: {rmse:.2f} bits")
        return rmse, mean_error_ratio
    else:
        print("No data to calculate RMSE.")
    # --- RMSE 계산 코드 종료 ---

