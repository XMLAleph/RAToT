#true = True
false = False

{
    "question_id": "q_10839", 
    "question_text": "In what country did Bain attend doctoral seminars of Wlad Godzich?", 
    "answers_objects": 
        [
            {
                "number": "", 
                "date": {"day": "", "month": "", "year": ""}, 
                "spans": ["Switzerland"]
            }
        ], 
    "contexts": 
        [
            {
                "title": "London", 
                "paragraph_text": "- London City Airport, the most central airport and the one with the shortest runway, in Newham, East London, is focused on business travellers, with a mixture of full-service short-haul scheduled flights and considerable business jet traffic.", 
                "is_supporting": false, 
                "idx": 0
            }, 
            {"title": "Les Baux-de-Provence", "paragraph_text": "- P. Destandau, Unpublished documents on the town of Baux, Vol. III, Memoirs of the Academy of Vaucluse, 1903", "is_supporting": false, "idx": 1}, 
            {"title": "Wlad Godzich", "paragraph_text": "Organizer of dozens of international conferences, he also acts as consultant to many university presses and organizers of university programs in the Americas and Europe. He sits on the editorial board of multiple American, European and Asian journals, both print and electronic. His research grants have been primarily from US, Canadian, Swedish, Swiss and private agencies.", "is_supporting": false, "idx": 2}, 
            {"title": "Paris Nanterre University", "paragraph_text": "The university is renowned in the fields of Law and Economics. Even though French universities are required by law to admit anyone with a Baccalaur\u00e9at, strain is put on the students from the start and the first year drop-out rate consistently hovers in the 60% region. At the postgraduate level, the university offers very competitive programs (highly selective master's degrees in Law and Business) and partnerships with some grandes \u00e9coles such as the Ecole Polytechnique, ESSEC, Ecole des Mines de Paris, and ESCP Europe among others.", "is_supporting": false, "idx": 3}, 
            {"title": "Salamanca", "paragraph_text": "Two of the largest businesses, both of them numbered among the largest 100 enterprises in the region, are the veterinary vaccine manufacturer \"Laboratorios Intervet\", and the fertilizer specialist manufacturers , which is the city's oldest industrial company, having been established originally as a starch factory in 1812.", "is_supporting": false, "idx": 4}, 
            {"title": "Classe pr\u00e9paratoire aux grandes \u00e9coles", "paragraph_text": "- TSI1, Physiques, Technologie, sciences industrielles (\"physics, technology, industrial science\") in the first year, followed by TSI2", "is_supporting": false, "idx": 5}, 
            {"title": "Dyslexia", "paragraph_text": "Dual route theory.The dual-route theory of reading aloud was first described in the early 1970s. This theory suggests that two separate mental mechanisms, or cognitive routes, are involved in reading aloud. One mechanism is the lexical route, which is the process whereby skilled readers can recognize known words by sight alone, through a \"dictionary\" lookup procedure. The other mechanism is the nonlexical or sublexical route, which is the process whereby the reader can \"sound out\" a written word. This is done by identifying the word's constituent parts (letters, phonemes, graphemes) and applying knowledge of how these parts are associated with each other, for example, how a string of neighboring letters sound together. The dual-route system could explain the different rates of dyslexia occurrence between different languages (e.g., the consistency of phonological rules in the Spanish language could account for the fact that Spanish-speaking children show a higher level of performance in non-word reading, when compared to English-speakers).", "is_supporting": false, "idx": 6}, 
            {"title": "Golo Mann", "paragraph_text": "Education.An average pupil, he received a classical education at the Wilhelms-Gymnasium in Munich beginning in September 1918, revealing talents in history, Latin, and especially in reciting poems, the latter being a lifelong passion. Increasingly sensing his parents\u2018 home as a burden, he attempted a kind of break-out by joining the Boy Scouts in spring 1921. On one of the holiday marches he was the victim of a slight sexual violation by his group leader.", "is_supporting": false, "idx": 7}, 
            {"title": "University of Geneva", "paragraph_text": "The University of Geneva (French: Universit\u00e9 de Gen\u00e8ve) is a public research university located in Geneva, Switzerland.", "is_supporting": true, "idx": 8}, 
            {"title": "Jules Ferry", "paragraph_text": "- 3 January 1885 \u2013 Jules Louis Lewal succeeds Campenon as Minister of War.", "is_supporting": false, "idx": 9}, 
            {"title": "Kingston upon Thames", "paragraph_text": "Kingston has been covered in literature, film and television. It is where the comic Victorian novel Three Men in a Boat by Jerome K. Jerome begins; cannons aimed against the Martians in H. G. Wells' The War of the Worlds are positioned on Kingston Hill; in The Rainbow by D. H. Lawrence the youngest Brangwen dreams of a job in Kingston upon Thames in a long, lyrical passage; Mr. Knightly in Emma by Jane Austen regularly visits Kingston, although the narrative never follows him there.", "is_supporting": false, "idx": 10}, 
            {"title": "Liphook", "paragraph_text": "Railways caused the long-distance coaching trade to reduce in the village. The railway station became the hub of short-distance horse-drawn transport, with the blacksmiths shop in The Square flourishing until at least 1918.", "is_supporting": false, "idx": 11}, 
            {"title": "University of Geneva", "paragraph_text": "Academic year.UNIGE's academic year runs from mid-September to mid-June. It is divided in two semesters, each one being concluded by an examination session, held respectively at the beginning of January and at the beginning of June. An examination session is held at the end of August and beginning of September as a retake for students who failed their January or June examinations.", "is_supporting": false, "idx": 12}], 
        "pinned_contexts": 
            # 从内容上看，这里的pinned_context应该算是一个gold文档，但为什么要和其他文档分开？？？？？上面的context中只有一条是相关文档，加上这里pinned_context这一条才能回答问题
            # 看下数据集原始论文是怎么设定的
            # 目前只有IIRC一个数据集有pinned context，看下dataset_reader里除了pinned context，其他部分的情况
            [
                {
                    "idx": 0, 
                    "title": "Thomas Bain (Orange)", 
                    "paragraph_text": "Bain was born in London. He lived Kingston upon Thames attending prep school at Highfield School (Liphook, Hampshire). He suffered from Dyslexia, and made slow progress in the educational system. In 1982 he moved to Spain, and took up Hispanic Studies in a small private college in Salamanca where he met up with friends of Golo Mann. Upon return to France he qualified for the Classe pr\u00e9paratoire aux grandes \u00e9coles. He accomplished his Kh\u00e2gne in the Lyce\u00e9 Jules Ferry. The same year he discovered a new archeological area at Les Baux-de-Provence. He accomplished his BA Humanities in the radical Paris Nanterre University. He completed M. Phil at the Geneva-based IUEE (Institute for European Studies), and later attended the doctoral seminars of Wlad Godzich in the University of Geneva.", 
                    "is_supporting": true
                }
            ], 
        "valid_titles": 
            ["London", "Kingston upon Thames", "Liphook", "Dyslexia", "Salamanca", "Golo Mann", "Classe pr\u00e9paratoire aux grandes \u00e9coles", "Jules Ferry", "Les Baux-de-Provence", "Paris Nanterre University", "Wlad Godzich", "University of Geneva"]
}