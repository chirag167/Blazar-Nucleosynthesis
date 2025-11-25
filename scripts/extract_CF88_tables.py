from utilities import extract_cf88_tables

def main():
    extract_cf88_tables(
        url="http://www.nuclear.csdb.cn/data/CF88/tables.html#008",
        name='H3(D,N)HE4',
        outdir="../data/CF88",
    )

if __name__ == "__main__":
    main()
