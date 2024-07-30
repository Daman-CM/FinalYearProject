import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial import distance
from networkx.algorithms import isomorphism
from pathlib import Path
import os
import time
import re


start_time = time.time()


def setup_paths():
    """
    Set up folder and result paths.
    """
    f_path = "C:\\Users\\daman\\OneDrive\\Documents\\Psychology Data Corpus"
    r_path = os.path.join(f_path, "Results")
    return f_path, r_path


def process_data_and_generate_graphs(f_path):
    """
    Process data files in the specified folder and generate dynamic graphs.
    """
    for filename in os.listdir(f_path):
        if filename.endswith(".OpIE"):
            opie_file_path = os.path.join(f_path, filename)
            opie_conversion(opie_file_path)
        elif filename.endswith(".RVB"):
            rvb_file_path = os.path.join(f_path, filename)
            rvb_conversion(rvb_file_path)

    csv_files = [file for file in os.listdir(f_path) if file.endswith('.csv')]

    for csv_file in csv_files:
        csv_path = os.path.join(f_path, csv_file)

        modify_csv(csv_path)
        create_gexf(csv_path)


def opie_conversion(opie_path):
    """
    Converts data from a tab-separated OpIE file into a DataFrame and saves it as a CSV file.
    """
    data_list = []

    with open(opie_path, 'r') as file:
        for line in file:
            words = line.strip().split(',')
            if len(words) == 3:
                word1, word2, word3 = words
                data_list.append({"NOUN": word1.strip(), "VERB/PREP": word2.strip(), "NOUN2": word3.strip()})

    df_opie = pd.DataFrame(data_list)

    output_csv_path = os.path.splitext(opie_path)[0] + '_OpIE_output.csv'
    df_opie.to_csv(output_csv_path, index=False)


def rvb_conversion(rvb_path):
    """
    Converts data from a tab-separated RVB file into a DataFrame and saves it as a CSV file.
    """
    data_list = []

    with open(rvb_path, 'r') as file:
        for line in file:
            elements = line.strip().split('\t')
            if line.strip() == '':
                data_list = []
            elif len(elements) >= 5:
                noun = elements[2].strip()
                verb_prep = elements[3].strip()
                noun2 = elements[4].strip()

                data_list.append({"NOUN": noun, "VERB/PREP": verb_prep, "NOUN2": noun2})

    df2 = pd.DataFrame(data_list)

    output_csv_path = os.path.splitext(rvb_path)[0] + '_RVB_output.csv'
    df2.to_csv(output_csv_path, index=False)


def modify_csv(csv_path):
    """
    Modify a CSV file by:
    1. Renaming second column containing 'NOUN' with 'NOUN2'.
    2. Adding a 'Time Start' column filled with integers starting from 1.
    3. Adding a 'Time End' column filled with the final value of 'Time Start' + 1.
    """
    try:
        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"Warning: CSV file '{csv_path}' is empty.")
            return

        noun_columns = [col for col in df.columns if 'NOUN' in col]
        for i, col in enumerate(noun_columns):
            new_col_name = f'NOUN{i + 1}' if i > 0 else 'NOUN'
            df = df.rename(columns={col: new_col_name})

        df['Time Start'] = range(1, len(df) + 1)

        if not df.empty:
            df['Time End'] = df['Time Start'].iloc[-1] + 1
        else:
            df['Time End'] = 0

        df.to_csv(csv_path, index=False)

    except pd.errors.EmptyDataError:
        print(f"Error: CSV file '{csv_path}' is empty.")


def create_gexf(csv_path):
    """Create a dynamic graph from a CSV file and save it in GEXF format."""
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Warning: CSV file '{csv_path}' is empty.")
            return

        g = nx.MultiDiGraph()

        for index, row in df.iterrows():
            source_noun = row['NOUN']
            target_noun2 = row['NOUN2']
            edge_verb_prep = row['VERB/PREP']
            timestamp_start = row['Time Start']
            timestamp_end = row['Time End']

            add_nodes_edges_to_graph(g, source_noun, target_noun2, edge_verb_prep, timestamp_start, timestamp_end)

        output_path = Path(csv_path).with_suffix(".gexf")
        gexf_path = output_path.parent / f"{output_path.stem}-dynamic_graph.gexf"
        nx.write_gexf(g, gexf_path)

    except pd.errors.EmptyDataError:
        print(f"Error: CSV file '{csv_path}' is empty.")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")


def add_nodes_edges_to_graph(g, source, target, edge, timestamp_start, timestamp_end):
    """Add nodes and edges to the graph with timestamps."""
    g.add_node(source, start=timestamp_start)
    g.add_node(target, start=timestamp_start)
    g.add_edge(source, target, label=edge, start=timestamp_start, end=timestamp_end, color='None')


def read_static_gexf(file_path):
    try:
        # Attempt to read the GEXF file
        g = nx.read_gexf(file_path, relabel=True)
        return g
    except Exception as e:

        print(f"Error reading GEXF file {file_path}: {e}")
        return nx.Graph()  # Return an empty graph in case of an error


def metric_operations(fr_path, res_path):
    """
    Perform metric operations.
    """
    num_nodes_difference(fr_path)
    centrality_difference(fr_path)
    density_difference(fr_path)
    pagerank_difference(fr_path)
    vector_construction(res_path)
    truncating(res_path)


def num_nodes_difference(path):
    """
    Calculate the difference in the number of nodes between each pair of graphs in the given directory.

    Generates:
        1. 'Metric_num_of_nodes.csv': Contains the differences in the number of nodes between each pair of graphs.
        2. 'Metric_num_of_nodes_times.csv': Contains the runtime for each iteration and the mean rank position of
         the smallest difference.

    """
    files = [f for f in os.listdir(path) if f.endswith('.gexf')]

    all_differences = []
    runtimes = []
    mean_positions = []
    std_positions = []
    total_time = 0

    for i in range(len(files)):
        st_time = time.time()

        differences = []

        file1 = os.path.join(path, files[i])
        graph1 = read_static_gexf(file1)
        num_nodes1 = graph1.number_of_nodes()

        differences.append(os.path.basename(file1))

        for j in range(0, len(files)):
            file2 = os.path.join(path, files[j])

            graph2 = read_static_gexf(file2)
            num_nodes2 = graph2.number_of_nodes()

            difference = abs(num_nodes1 - num_nodes2)

            differences.append((difference, os.path.basename(file2)))

        differences[1:] = sorted(differences[1:], key=lambda x: x[0])

        pattern = re.compile(r'^(\d+ [a-zA-Z]+(?: -?\d* [\da-zA-Z-]+)?)')

        first_filename = differences[0]
        match = pattern.match(first_filename)
        if match:
            substring = match.group(1)

            positions = []

            for index, (_, filename) in enumerate(differences[1:], start=1):
                if substring in filename:
                    positions.append(index)

            if positions:
                mean_position = np.mean(positions)
                std_position = np.std(positions)
                mean_positions.append(mean_position)
                std_positions.append(std_position)
            else:
                mean_positions.append(np.nan)
                std_positions.append(np.nan)
        else:
            mean_positions.append(np.nan)
            std_positions.append(np.nan)

        all_differences.append(differences)

        end_tie = time.time()
        runtime = end_tie - st_time

        total_time += runtime

        runtimes.append(runtime)

    df = pd.DataFrame(all_differences)

    specified_columns_df = pd.DataFrame()

    specified_columns_df['Loop Time (sec)'] = runtimes

    specified_columns_df['Rank Mean'] = mean_positions
    specified_columns_df['Standard Deviation'] = std_positions

    specified_columns_df.to_csv(os.path.join(path, 'Times', 'Metric_num_of_nodes_times.csv'), index=False)

    column_labels_main = ['Reference'] + [f'Rank {i}' for i in range(1, len(df.columns))]

    df.columns = column_labels_main

    df.to_csv(os.path.join(path, 'Results', 'Metric_num_of_nodes.csv'), index=False)


def centrality_difference(path):
    """
    Calculate centrality differences between graphs in a directory and store results in CSV files.

    The method computes centrality differences between each pair of graphs in the directory.
    It calculates the cosine similarity between centrality vectors of nodes in the graphs.
    For each graph, it determines the rank of most similar graphs and computes mean and standard deviation of ranks.
    Finally, it saves the results to CSV files in the 'Results' and 'Times' directories within the given path.
    """
    files = [f for f in os.listdir(path) if f.endswith('.gexf')]

    all_cen = []
    runtimes = []
    mean_positions = []
    std_positions = []
    total_time = 0

    for i in range(len(files)):
        st_time = time.time()

        dissimilarities = []

        file1 = os.path.join(path, files[i])
        graph1 = read_static_gexf(file1)
        centrality1 = nx.degree_centrality(graph1)

        dissimilarities.append(os.path.basename(file1))

        for j in range(0, len(files)):
            file2 = os.path.join(path, files[j])

            graph2 = read_static_gexf(file2)
            centrality2 = nx.degree_centrality(graph2)
            centrality1_vector = np.array(list(centrality1.values()))
            centrality2_vector = np.array(list(centrality2.values()))

            if len(centrality1_vector) < len(centrality2_vector):
                centrality1_vector = np.pad(centrality1_vector, (0, len(centrality2_vector) - len(centrality1_vector)))
            elif len(centrality1_vector) > len(centrality2_vector):
                centrality2_vector = np.pad(centrality2_vector, (0, len(centrality1_vector) - len(centrality2_vector)))

            cosine_similarity = 1 - distance.cosine(centrality1_vector, centrality2_vector)
            dissimilarities.append((cosine_similarity, os.path.basename(file2)))

        dissimilarities[1:] = sorted(dissimilarities[1:], key=lambda x: x[0], reverse=True)

        pattern = re.compile(r'^(\d+ [a-zA-Z]+(?: -?\d* [\da-zA-Z-]+)?)')

        first_filename = dissimilarities[0]
        match = pattern.match(first_filename)
        if match:
            substring = match.group(1)

            positions = []

            for index, (_, filename) in enumerate(dissimilarities[1:], start=1):
                if substring in filename:
                    positions.append(index)

            if positions:
                mean_position = np.mean(positions)
                std_position = np.std(positions)
                mean_positions.append(mean_position)
                std_positions.append(std_position)
            else:
                mean_positions.append(np.nan)
                std_positions.append(np.nan)
        else:
            mean_positions.append(np.nan)
            std_positions.append(np.nan)

        all_cen.append(dissimilarities)

        end_t = time.time()
        runtime = end_t - st_time

        total_time += runtime

        runtimes.append(runtime)
    df = pd.DataFrame(all_cen)

    specified_columns_df = pd.DataFrame()
    specified_columns_df['Loop Time (sec)'] = runtimes
    specified_columns_df['Rank Mean'] = mean_positions
    specified_columns_df['Standard Deviation'] = std_positions
    specified_columns_df.to_csv(os.path.join(path, 'Times', 'Metric_centrality_times.csv'), index=False)

    column_labels_main = ['Reference'] + [f'Rank {i}' for i in range(1, len(df.columns))]
    df.columns = column_labels_main
    df.to_csv(os.path.join(path, 'Results', 'Metric_centrality.csv'), index=False)


def density_difference(path):
    """
    Calculate density difference between graphs in a given directory.

    This function reads graph files in the specified directory, computes the density for each graph,
    and calculates the density difference between each pair of graphs. It then saves the results to CSV files.

    """
    files = [f for f in os.listdir(path) if f.endswith('.gexf')]

    all_den = []
    runtimes = []
    mean_positions = []
    std_positions = []
    total_time = 0

    for i in range(len(files)):
        st_time = time.time()
        disparities = []
        file1 = os.path.join(path, files[i])
        graph1 = read_static_gexf(file1)
        density1 = nx.density(graph1)

        disparities.append(os.path.basename(file1))

        for j in range(0, len(files)):
            file2 = os.path.join(path, files[j])

            graph2 = read_static_gexf(file2)
            density2 = nx.density(graph2)

            density_differences = abs(density1 - density2)

            disparities.append((density_differences, os.path.basename(file2)))

        disparities[1:] = sorted(disparities[1:], key=lambda x: x[0])

        pattern = re.compile(r'^(\d+ [a-zA-Z]+(?: -?\d* [\da-zA-Z-]+)?)')

        first_filename = disparities[0]
        match = pattern.match(first_filename)
        if match:
            substring = match.group(1)

            positions = []

            for index, (_, filename) in enumerate(disparities[1:], start=1):
                # Check if the filename contains the substring pattern
                if substring in filename:
                    positions.append(index)

            if positions:
                mean_position = np.mean(positions)
                std_position = np.std(positions)
                mean_positions.append(mean_position)
                std_positions.append(std_position)
            else:
                mean_positions.append(np.nan)
                std_positions.append(np.nan)
        else:
            mean_positions.append(np.nan)
            std_positions.append(np.nan)

        all_den.append(disparities)

        end_t = time.time()
        runtime = end_t - st_time

        total_time += runtime

        runtimes.append(runtime)

    df = pd.DataFrame(all_den)

    specified_columns_df = pd.DataFrame()
    specified_columns_df['Loop Time (sec)'] = runtimes
    specified_columns_df['Rank Mean'] = mean_positions
    specified_columns_df['Standard Deviation'] = std_positions
    specified_columns_df.to_csv(os.path.join(path, 'Times', 'Metric_density_times.csv'), index=False)

    column_labels_main = ['Reference'] + [f'Rank {i}' for i in range(1, len(df.columns))]
    df.columns = column_labels_main
    df.to_csv(os.path.join(path, 'Results', 'Metric_density.csv'), index=False)


def pagerank_difference(path):
    """
    Calculates the difference in PageRank values for each pair of graphs in the specified directory.
    For each graph, it computes the difference in the sum of PageRank values between itself and every other graph.
    It then calculates the mean position and standard deviation of the graph in the sorted order of
    differences among the graphs.
    Finally, it saves the computed metrics into CSV files.

    """
    files = [f for f in os.listdir(path) if f.endswith('.gexf')]

    all_pages = []
    total_time = 0
    runtimes = []
    mean_positions = []
    std_positions = []

    for i in range(len(files)):
        sts_time = time.time()
        variances = []

        file1 = os.path.join(path, files[i])
        graph1 = read_static_gexf(file1)
        pagerank1 = nx.pagerank(graph1)

        variances.append(os.path.basename(file1))

        for j in range(0, len(files)):
            file2 = os.path.join(path, files[j])
            graph2 = read_static_gexf(file2)
            pagerank2 = nx.pagerank(graph2)

            difference = abs(sum(pagerank1.values()) - sum(pagerank2.values()))
            variances.append((difference, os.path.basename(file2)))

        variances[1:] = sorted(variances[1:], key=lambda x: x[0])

        pattern = re.compile(r'^(\d+ [a-zA-Z]+(?: -?\d* [\da-zA-Z-]+)?)')
        first_filename = variances[0]
        match = pattern.match(first_filename)

        if match:
            substring = match.group(1)
            positions = []

            for index, (_, filename) in enumerate(variances[1:], start=1):
                if substring in filename:
                    positions.append(index)

            if positions:
                mean_position = np.mean(positions)
                std_position = np.std(positions)
                mean_positions.append(mean_position)
                std_positions.append(std_position)
            else:
                mean_positions.append(np.nan)
                std_positions.append(np.nan)
        else:
            mean_positions.append(np.nan)
            std_positions.append(np.nan)

        all_pages.append(variances)

        ends_time = time.time()
        runtime = ends_time - sts_time
        total_time += runtime
        runtimes.append(runtime)
    df = pd.DataFrame(all_pages)

    specified_columns_df = pd.DataFrame()
    specified_columns_df['Loop Time (sec)'] = runtimes
    specified_columns_df['Rank Mean'] = mean_positions
    specified_columns_df['Standard Deviation'] = std_positions
    specified_columns_df.to_csv(os.path.join(path, 'Times', 'Metric_pagerank_times.csv'), index=False)

    column_labels_main = ['Reference'] + [f'Rank {i}' for i in range(1, len(df.columns))]
    df.columns = column_labels_main
    df.to_csv(os.path.join(path, 'Results', 'Metric_pagerank.csv'), index=False)


def vector_construction(directory_path):
    """
    Construct a vector of metrics from Metric CSV files in the specified directory.

    This function reads CSV files from the given directory, extracts rank information,
    and constructs a vector of metrics based on the extracted data. The resulting
    vector is saved as 'Vector of Metrics.csv' in the same directory.

    """
    vector_metrics_file = os.path.join(directory_path, 'Vector of Metrics.csv')
    if os.path.exists(vector_metrics_file):
        try:
            os.remove(vector_metrics_file)
        except Exception as e:
            print(f"Error occurred while deleting existing file: {e}")

    csv_files = [file for file in os.listdir(directory_path)
                 if file.endswith('.csv')
                 and file != 'Pre-Truncation Results.csv'
                 and file != 'Post-Truncation Result.cs']

    combined_data = {}
    for file in csv_files:
        df = pd.read_csv(os.path.join(directory_path, file))

        num_rank_columns = sum(col.startswith('Rank') for col in df.columns)

        total_columns = num_rank_columns + 1

        rank_columns = [f'Rank {i}' for i in range(1, total_columns)]

        for index, row in df.iloc[1:total_columns].iterrows():
            for col in rank_columns:
                try:
                    rank_info = row[col]
                    if not pd.isna(rank_info):
                        filename = rank_info.split(',')[1][1:-1]
                        rank_number = int(col.split()[1])
                        vector = (filename, rank_number)

                        reference = row['Reference']

                        if reference not in combined_data:
                            combined_data[reference] = []

                        updated = False
                        for existing_vector in combined_data[reference]:
                            if existing_vector[0] == filename:
                                combined_data[reference][combined_data[reference].index(existing_vector)] = (
                                    filename, *existing_vector[1:], rank_number)
                                updated = True
                                break

                        if not updated:
                            combined_data[reference].append(vector)

                except AttributeError as e:
                    print(
                        f"Error occurred in file '{file}' at row {index + 2}, column '{col}': {e}")
                    print(f"Row data: {row}")

    combined_list = [{'Reference': key, **{f'Vector {i + 1}': val for i, val in enumerate(values)}} for key, values
                     in combined_data.items()]

    combined_df = pd.DataFrame(combined_list)
    combined_df.to_csv(os.path.join(directory_path, 'Vector of Metrics.csv'), index=False)


def extract_filename(tuple_data):
    return tuple_data[0]


def process_tuple(tuple_data):
    filename = tuple_data[0]
    numbers = tuple_data[1:]
    processed_numbers = [float(num) * (1/4) for num in numbers]
    new_number = sum(processed_numbers)
    return filename, new_number


def truncating(folder_location):
    """

    Process and truncate data from 'Vector of Metrics.csv' in the specified folder location.
    The processed data is saved to 'Pre-Truncation Results.csv' while a truncated list is
    saved to 'Post-Truncation Result.csv'.

    """
    input_file_path = os.path.join(folder_location, 'Vector of Metrics.csv')
    output_file_path = os.path.join(folder_location, 'Pre-Truncation Results.csv')
    copy_file_path = os.path.join(folder_location, 'Post-Truncation Result.csv')

    if not os.path.exists(input_file_path):
        print(f"File 'Vector of Metrics.csv' does not exist in '{folder_location}'.")
        return

    df = pd.read_csv(input_file_path)

    for row_index in range(len(df)):
        row_data = df.iloc[row_index, 1:]
        row_data = [eval(value) for value in row_data]
        new_tuples = [process_tuple(tuple_data) for tuple_data in row_data]
        new_tuples.sort(key=lambda x: x[1])
        sorted_tuples = [(tup[0],) + (tup[1],) for tup in new_tuples]
        df.iloc[row_index, 1:] = sorted_tuples

    df.to_csv(output_file_path, index=False)

    df.iloc[:, :11].to_csv(copy_file_path, index=False)

    copy_df = pd.read_csv(copy_file_path)

    for row_index in range(1, copy_df.shape[0]):
        for col_index in range(1, copy_df.shape[1]):
            value = copy_df.iloc[row_index, col_index]
            if isinstance(value, str):
                filename = extract_filename(eval(value))
                copy_df.iloc[row_index, col_index] = filename

    copy_df.to_csv(copy_file_path, index=False)

    print(f"Pre-Truncation completed. Pre-Results saved to '{output_file_path}'.")
    print(f"Truncated Results saved to '{copy_file_path}' with filenames extracted.")


if __name__ == "__main__":
    folder_path, result_path = setup_paths()

    process_data_and_generate_graphs(folder_path)
    metric_operations(folder_path, result_path)


    end_time = time.time()

    elapsed_time = end_time - start_time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
