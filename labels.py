import pandas as pd


file_path = '30550_prod.csv'
df = pd.read_csv(file_path)


def assign_responder_labels(df):
    patients = sorted(set(col.split('_')[0] for col in df.columns if '_' in col), key=int)

    labels = []

    for patient in patients:
        patient_columns = sorted([col for col in df.columns if col.startswith(f"{patient}_")], key=lambda x: int(x.split('_')[1]))

        first_time_point = patient_columns[0]
        labels.append({'Patient': patient, 'TimePoint': first_time_point, 'Label': 0})

        for i in range(1, len(patient_columns)):

            curr_col = patient_columns[i]

            integrated_value_curr = df[curr_col].sum()

            tipping_point = False
            for j in range(i):
                prev_col = patient_columns[j]
                integrated_value_prev = df[prev_col].sum()

                if integrated_value_curr >= 30 * integrated_value_prev:
                    tipping_point = True
                    break

            label = 1 if tipping_point else 0
            labels.append({'Patient': patient, 'TimePoint': curr_col, 'Label': label})

    label_df = pd.DataFrame(labels)

    return label_df


label_df = assign_responder_labels(df)

output_file_path = 'labels.csv'
label_df.to_csv(output_file_path, index=False)

print(f"Labels saved to {output_file_path}")