import os
from data.wikielements_loader import WikiElementsDataSet
from data.wikicities_loader import WikiCitiesDataSet
from data.wiki50_loader import Wiki50DataSet
from data.wikisection_loader import WikiSectioniDataSet
from data.clinical_loader import ClinicalDataSet
from data.manifesto_loader import ManifestoDataSet
from data.choi_loader import ChoiDataSet
from data.wiki_loader import WikipediaDataSet
def get_loader(args, encoder, data_type,split=None):
    if data_type == "wikielements":
        assert split == None
        dataset = WikiElementsDataSet(
            root = args.wikielements_path,
            args = args,
            encoder = encoder,
            length_filter = args.length_filter,
            sent_num_filter = args.sent_num_filter,
            local_rank = args.local_rank
        )
    elif data_type == "wikicities":
        assert split == None
        dataset = WikiCitiesDataSet(
            root = args.wikicities_path,
            args = args,
            encoder = encoder,
            length_filter = args.length_filter,
            sent_num_filter = args.sent_num_filter,
            local_rank = args.local_rank
        )
    elif data_type == "wiki50":
        assert split == None
        dataset = Wiki50DataSet(
            root = args.wiki50_path,
            args = args,
            encoder = encoder,
            length_filter = args.length_filter,
            sent_num_filter = args.sent_num_filter,
            local_rank = args.local_rank
        )
    elif data_type == "wikisection":
        assert split in ['train','dev','test']
        split = "validation" if split == 'dev' else split
        dataset = WikiSectioniDataSet(
            root = args.wikisection_path,
            args = args,
            encoder = encoder,
            split = split,
            length_filter = args.length_filter,
            sent_num_filter = args.sent_num_filter,
            local_rank = args.local_rank
        )
    elif data_type == "clinical":
        assert split == None
        dataset = ClinicalDataSet(
            root = args.clinical_path,
            args = args,
            encoder = encoder,
            length_filter = args.length_filter,
            sent_num_filter = args.sent_num_filter,
            local_rank = args.local_rank
        )
    elif data_type == "manifesto":
        assert split == None
        dataset = ManifestoDataSet(
            root = args.manifesto_path,
            args = args,
            encoder = encoder,
            length_filter = args.length_filter,
            sent_num_filter = args.sent_num_filter,
            local_rank = args.local_rank
        )
    elif data_type == "choi":
        assert split in [None, 'train', 'dev', 'test']
        dataset = ChoiDataSet(
            root = args.choi_path,
            args = args,
            encoder = encoder,
            split=split,
            length_filter = args.length_filter,
            sent_num_filter = args.sent_num_filter,
            local_rank = args.local_rank
        )
    elif data_type == "wiki":
        assert split in ['train', 'dev', 'test']
        dataset = WikipediaDataSet(
            args = args,
            root = args.wiki_path,
            split = split,
            encoder = encoder,
            length_filter = args.length_filter,
            sent_num_filter = args.sent_num_filter,
            data_frac = args.data_frac,
            high_granularity=args.high_granularity,
            local_rank=args.local_rank
        )
    else:
        print(f"{data_type} is false data type")
        return None
    return dataset


if __name__ == "__main__":
    from parameters import *
    args = create_parser()
    logger = setup_logger(args.logger_name, os.path.join(args.checkpoint_dir, 'train.log'))
    data_type = "choi"
    split = None
    dataset = get_loader(args, None, data_type,split=split)