import csv


def merge_csv():
    data_rows_csv1 = []
    data_rows_csv2 = []

    with open('IIWA14_extended_nolinear/data/local/experiment/meta3_rl2_ppo/progress.csv', 'r') as file:
        reader = csv.reader(file)
        header_master = next(reader)
        for data_row in reader:
            data_rows_csv1.append(data_row)

    with open('IIWA14_extended_nolinear/data/local/experiment/meta3_rl2_ppo/pre_trained_meta3_rl2_ppo_1/progress.csv',
              'r') as file:
        reader = csv.reader(file)
        header_merge = next(reader)
        for data_row in reader:
            data_rows_csv2.append(data_row)

    # MetaTest ist not logged during resume of the experiment -> get information from debug.log
    text_file = open('IIWA14_extended_nolinear/data/local/experiment/meta3_rl2_ppo/pre_trained_meta3_rl2_ppo_1/debug.log', 'r')
    file_lines = text_file.readlines()

    # Extract MetaTest data
    compare = [data for data in file_lines if 'MetaTest/' in data]

    iterations_master = [line[header_master.index('Average/Iteration')] for line in data_rows_csv1]
    iterations_merge = [line[header_merge.index('Average/Iteration')] for line in data_rows_csv2]

    if (int(iterations_master[-1]) + 1) != int(iterations_merge[0]):
        cut_elements_master = int(iterations_merge[0])
        data_rows_csv1 = data_rows_csv1[0:cut_elements_master]

    separated_strings = [data.split() for data in compare]
    meta_test_strings = [elem[0] for elem in separated_strings]
    meta_test_strings = meta_test_strings[0:27]
    fill_string = 27 * ['']

    for i, string in enumerate(meta_test_strings):
        header_merge.append(string)
        count_similar_rows = 0
        for j, _ in enumerate(data_rows_csv2):
            insert_active = False
            if j != 0:
                if ((j+1) % 10) == 0:
                    elem = separated_strings[i + (count_similar_rows * 27)]
                    data_rows_csv2[j].append(elem[1])
                    count_similar_rows += 1
                    insert_active = True
            if i == 0 and not insert_active:
                data_rows_csv2[j].extend(fill_string)

    index_list = []
    for element in header_master:
        index_list.append(header_merge.index(element))

    reordered_single_row_csv2 = []
    reordered_data_rows_csv2 = []
    for row in data_rows_csv2:
        for elem in index_list:
            reordered_single_row_csv2.append(row[elem])
        reordered_data_rows_csv2.append(reordered_single_row_csv2)
        reordered_single_row_csv2 = []

    reordered_header_merge = []
    for elem in index_list:
        reordered_header_merge.append(header_merge[elem])

    data_rows_csv1.extend(reordered_data_rows_csv2)

    final_csv_data = []
    final_csv_data.append(header_master)
    final_csv_data.extend(data_rows_csv1)

    with open('progress.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(final_csv_data)

    file.close()

if __name__ == "__main__":
    merge_csv()