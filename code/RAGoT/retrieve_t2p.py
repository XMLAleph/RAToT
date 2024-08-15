from Retriever.ela_retriever import UnifiedRetriever
import argparse
# 看下这个代码能不能实现从corpus根据题目检索文本

def parse_arguments():
    # python retrieve_t2p.py --corpus 2wikimultihopqa --title The Girl in the Taxi (1937 film)
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument("--corpus", type=str, required=False, help="input dataset file")
    #arg_parser.add_argument("--title", type=str, required=True, help="paragraph title")
    return arg_parser.parse_args()

def main():
    args = parse_arguments()
    corpus = args.corpus
    title = ["The Girl in Number 29"]
    '''
    if isinstance(args.title, list[str]):
        print("one title one time")
        title = args.title[0]
    '''  
    retriever = UnifiedRetriever()
    
    params = {
        "query_text": "",
        "max_hits_count": 1,
        "corpus_name": corpus,
        "allowed_titles":title,
    }
    
    result = retriever.retrieve_from_elasticsearch(**params)
    
    print(result)
    
if __name__ == "__main__":
    main()