import csv

def create_csv_file(filepath, columns, delimiter=',', mode='w+'):
    """
    args:
        file_path: path of the file to be created
        columns: column names for the prefix
        delimiter
        mode

    returns:
        csv_out: file handler in order to close the file later in the program
        writer: writer to write rows into the csv file
    """
    try:
        csv_out = open(filepath, mode=mode)
        writer = csv.writer(csv_out, delimiter=delimiter)
        writer.writerow(columns)
        return csv_out, writer
    except OSError:
        print("An error occured while attempting to create/write csv file.")
        exit(1)