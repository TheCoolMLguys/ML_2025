import os
import csv
import pandas as pd
from mondrian import mondrian


class GlobalAnonymizer:
    def __init__(self, k=5, strict=True):
        self.k = k
        self.strict = strict
        self.result = None
        self.eval_result = None

    def anonymize(self, data, qi_len=-1):
        print(f"Data size: {len(data)} ")
        print(f"Number of QI: {qi_len}")

        self.result, self.eval_result = mondrian(
            data,
            self.k,
            relax=not self.strict,
            QI_num=qi_len
        )

        print(f"After anonymization, NCP: {self.eval_result[0]:.2f}%, Time: {self.eval_result[1]:.2f}s")
        return self.result, self.eval_result


def prepare_student_placement_data(file_path='Exercise3/data/student_placement.csv'):
    print("Preparing Student Placement dataset...")


    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    original_columns = list(df.columns)
    qi_columns = [col for col in original_columns if col not in ['Student_ID', 'Placement_Status']]
    sa_column = 'Placement_Status'

    print(f"\nQI columns ({len(qi_columns)}): {qi_columns}")
    print(f"SA column: {sa_column}")

    config = {
        'name': 'student_placement',
        'file_path': file_path,
        'qi_columns': qi_columns,
        'sa_column': sa_column,
        'is_categorical': [
            False,  # Age
            True,  # Gender
            True,  # Degree
            True,  # Branch
            False,  # CGPA
            False,  # Internships
            False,  # Projects
            False,  # Coding_Skills
            False,  # Communication_Skills
            False,  # Aptitude_Test_Score
            False,  # Soft_Skills_Rating
            False,  # Certifications
            False,  # Backlogs
        ],
        'k': 5,
        'strict': True,
        'has_header': True,
        'delimiter': ','
    }

    data = []
    intuitive_dicts = []
    intuitive_order = []
    intuitive_numbers = []

    for i in range(len(qi_columns)):
        intuitive_dicts.append({})
        intuitive_numbers.append(0)
        intuitive_order.append([])

    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')

        next(reader)

        for row_num, row in enumerate(reader):
            if not row or len(row) < 15:
                continue

            processed_row = []

            for i, col_name in enumerate(qi_columns):
                col_index = original_columns.index(col_name)
                value = row[col_index].strip()

                if config['is_categorical'][i]:
                    if value in intuitive_dicts[i]:
                        processed_row.append(intuitive_dicts[i][value])
                    else:
                        intuitive_dicts[i][value] = intuitive_numbers[i]
                        processed_row.append(intuitive_numbers[i])
                        intuitive_order[i].append(value)
                        intuitive_numbers[i] += 1
                else:
                    try:
                        if '.' in value:
                            processed_row.append(float(value))
                        else:
                            processed_row.append(int(value))
                    except ValueError:
                        if value in intuitive_dicts[i]:
                            processed_row.append(intuitive_dicts[i][value])
                        else:
                            intuitive_dicts[i][value] = intuitive_numbers[i]
                            processed_row.append(intuitive_numbers[i])
                            intuitive_order[i].append(value)
                            intuitive_numbers[i] += 1

            sa_index = original_columns.index(sa_column)
            processed_row.append(row[sa_index].strip())

            data.append(processed_row)

    return data, intuitive_order, config


def prepare_breast_cancer_data(file_path='Exercise3/data/breast-cancer-diagnostic.shuf.lrn.csv'):
    print("\nPreparing Breast Cancer Diagnostic dataset...")

    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    original_columns = list(df.columns)
    qi_columns = [col for col in original_columns if col not in ['ID', 'class']]
    sa_column = 'class'

    print(f"\nQI columns ({len(qi_columns)}): First 5: {qi_columns[:5]}...")
    print(f"SA column: {sa_column}")

    config = {
        'name': 'breast_cancer',
        'file_path': file_path,
        'qi_columns': qi_columns,
        'sa_column': sa_column,
        'is_categorical': [False] * 30,
        'k': 5,
        'strict': True,
        'has_header': True,
        'delimiter': ','
    }

    data = []
    intuitive_order = [[] for _ in range(len(qi_columns))]

    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')

        next(reader)

        for row_num, row in enumerate(reader):
            if not row or len(row) < 32:
                continue

            processed_row = []

            for i, col_name in enumerate(qi_columns):
                col_index = original_columns.index(col_name)
                value = row[col_index].strip()
                try:
                    processed_row.append(float(value))
                except ValueError:
                    processed_row.append(0.0)

            sa_index = original_columns.index(sa_column)
            processed_row.append(row[sa_index].strip())

            data.append(processed_row)

    return data, intuitive_order, config


def prepare_teen_phone_data(file_path='Exercise3/data/teen_phone_addiction_dataset.csv'):
    print("\nPreparing Teen Phone Addiction dataset...")

    df = pd.read_csv(file_path)
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    original_columns = list(df.columns)
    identifiers = ['ID', 'Name', 'Location']
    qi_columns = [col for col in original_columns if col not in identifiers + ['Addiction_Level']]
    sa_column = 'Addiction_Level'

    print(f"\nQI columns ({len(qi_columns)}): {qi_columns}")
    print(f"SA column: {sa_column}")

    is_categorical = []
    for col in qi_columns:
        unique_vals = df[col].nunique()
        if col in ['Gender', 'Phone_Usage_Purpose']:
            is_categorical.append(True)
        elif unique_vals < 10 and col not in ['Age', 'School_Grade']:
            is_categorical.append(True)
        else:
            is_categorical.append(False)

    config = {
        'name': 'teen_phone',
        'file_path': file_path,
        'qi_columns': qi_columns,
        'sa_column': sa_column,
        'is_categorical': is_categorical,
        'k': 5,
        'strict': True,
        'has_header': True,
        'delimiter': ','
    }

    data = []
    intuitive_dicts = []
    intuitive_order = []
    intuitive_numbers = []

    for i in range(len(qi_columns)):
        intuitive_dicts.append({})
        intuitive_numbers.append(0)
        intuitive_order.append([])

    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')

        next(reader)

        for row_num, row in enumerate(reader):
            if not row or len(row) < 25:
                continue

            processed_row = []

            for i, col_name in enumerate(qi_columns):
                col_index = original_columns.index(col_name)
                value = row[col_index].strip()

                if config['is_categorical'][i]:
                    if value in intuitive_dicts[i]:
                        processed_row.append(intuitive_dicts[i][value])
                    else:
                        intuitive_dicts[i][value] = intuitive_numbers[i]
                        processed_row.append(intuitive_numbers[i])
                        intuitive_order[i].append(value)
                        intuitive_numbers[i] += 1
                else:
                    try:
                        if col_name == 'School_Grade':
                            grade_str = ''.join(filter(str.isdigit, value))
                            if grade_str:
                                processed_row.append(int(grade_str))
                            else:
                                processed_row.append(0)
                        else:
                            processed_row.append(float(value))
                    except ValueError:
                        if value in intuitive_dicts[i]:
                            processed_row.append(intuitive_dicts[i][value])
                        else:
                            intuitive_dicts[i][value] = intuitive_numbers[i]
                            processed_row.append(intuitive_numbers[i])
                            intuitive_order[i].append(value)
                            intuitive_numbers[i] += 1

            sa_index = original_columns.index(sa_column)
            processed_row.append(row[sa_index].strip())

            data.append(processed_row)

    return data, intuitive_order, config


def convert_back_to_original(anonymized_data, intuitive_order, delimiter='~'):
    converted_data = []

    for record in anonymized_data:
        converted_record = []

        for i in range(len(intuitive_order)):
            if len(intuitive_order[i]) > 0:
                cell_value = str(record[i])

                if delimiter in cell_value:
                    parts = cell_value.split(delimiter)
                    try:
                        start_idx = int(parts[0])
                        end_idx = int(parts[1])
                        original_values = []
                        for idx in range(start_idx, end_idx + 1):
                            if idx < len(intuitive_order[i]):
                                original_values.append(intuitive_order[i][idx])
                        converted_record.append(delimiter.join(original_values))
                    except (ValueError, IndexError):
                        converted_record.append(cell_value)
                else:
                    try:
                        idx = int(cell_value)
                        if idx < len(intuitive_order[i]):
                            converted_record.append(intuitive_order[i][idx])
                        else:
                            converted_record.append(cell_value)
                    except ValueError:
                        converted_record.append(cell_value)
            else:
                converted_record.append(str(record[i]))

        converted_record.append(str(record[-1]))
        converted_data.append(converted_record)

    return converted_data


def run_anonymization_for_dataset(dataset_name, k=5, output_dir='Exercise3/anonymized_results_global'):
    print(f"\n{'=' * 60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'=' * 60}")

    os.makedirs(output_dir, exist_ok=True)

    if dataset_name == 'student_placement':
        data, intuitive_order, config = prepare_student_placement_data()
    elif dataset_name == 'breast_cancer':
        data, intuitive_order, config = prepare_breast_cancer_data()
    elif dataset_name == 'teen_phone':
        data, intuitive_order, config = prepare_teen_phone_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    anonymizer = GlobalAnonymizer(k=k, strict=True)

    anonymized_data, eval_metrics = anonymizer.anonymize(
        data,
        qi_len=len(config['qi_columns'])
    )

    print("\nConverting back to original values...")
    final_data = convert_back_to_original(anonymized_data, intuitive_order)

    output_file = os.path.join(output_dir, f"{config['name']}_global_k{k}.csv")

    # header with original column names
    with open(output_file, 'w') as f:
        header = config['qi_columns'] + [config['sa_column']]
        f.write(';'.join(header) + '\n')

        for record in final_data:
            f.write(';'.join(record) + '\n')


    print(f"\nResults for {config['name']}:")
    print(f"  NCP (Normalized Certainty Penalty): {eval_metrics[0]:.2f}%")
    print(f"  Runtime: {eval_metrics[1]:.2f} seconds")
    print(f"  Output saved to: {output_file}")
    print(f"  QI columns: {len(config['qi_columns'])}")
    print(f"  Sensitive attribute: {config['sa_column']}")


    return {
        'dataset': config['name'],
        'k': k,
        'ncp': eval_metrics[0],
        'runtime': eval_metrics[1],
        'output_file': output_file,
        'num_records': len(data),
        'num_qi': len(config['qi_columns']),
        'qi_columns': config['qi_columns'],
        'sa_column': config['sa_column']
    }


def main():
    datasets = ['student_placement', 'breast_cancer', 'teen_phone']

    all_results = []

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        print("-" * 40)

        try:
            if dataset == 'student_placement':
                data_path = 'Exercise3/data/student_placement.csv'
            elif dataset == 'breast_cancer':
                data_path = 'Exercise3/data/breast-cancer-diagnostic.shuf.lrn.csv'
            elif dataset == 'teen_phone':
                data_path = 'Exercise3/data/teen_phone_addiction_dataset.csv'

            if not os.path.exists(data_path):
                print(f"  Error: Data file not found at {data_path}")
                continue

            print(f"  Found data file: {data_path}")

            results = run_anonymization_for_dataset(dataset, k=5)
            all_results.append(results)

            print(f"Completed!")

        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    if all_results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(f"\n{'Dataset':<25} {'NCP (%)':<12} {'Time (s)':<12} {'Records':<10} {'QI Attr':<8} {'SA Column':<20}")
        print("-" * 90)

        total_ncp = 0
        total_time = 0
        total_records = 0

        for result in all_results:
            print(f"{result['dataset']:<25} {result['ncp']:<12.2f} {result['runtime']:<12.2f} "
                  f"{result['num_records']:<10} {result['num_qi']:<8} {result['sa_column']:<20}")
            total_ncp += result['ncp']
            total_time += result['runtime']
            total_records += result['num_records']



if __name__ == '__main__':
    main()