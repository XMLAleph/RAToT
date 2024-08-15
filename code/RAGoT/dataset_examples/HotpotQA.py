#true = True
false = False

{
    "dataset": "hotpotqa", 
    "question_id": "5a8b57f25542995d1e6f1371", 
    "question_text": "Were Scott Derrickson and Ed Wood of the same nationality?", 
    "level": "hard", 
    "type": "comparison", #问题的类型，这个问题是”比较型“的，不同类型的问题可能是对应的思维框架不一样
    "answers_objects": #问题的答案。从这里来看，答案包括三种类型：数字、日期、文本（比如”是/否“或“xxx”就是文本类型的答案
        [
            {
                "number": "", 
                "date": {"day": "", "month": "", "year": ""}, 
                "spans": ["yes"]
            }
        ],
    "contexts": #这里应该是数据集本身给出的参考性文本。
    # 但问题是为什么还要给出多个无关文本？？？
    # 问题在于搞清楚这里的文本是哪种类型（相关，相似，无关），只要不是相似文本就没什么影响
    # 事实情况是给出的context包括的就是相关文档和“相似文档”，也是检索时可能被找到的文档
    # 所以其他模型到底是怎么用这些文档的？？？怎么处理其中的相似文档的？
    # 相似文档就是干扰性文档
    # 这一点比较麻烦
    # 注意下ircot是怎么处理这些干扰性文档的
    # 之前也确实有工作探讨过处理相似文档的方法。不过在ircot的基础上改进的文章至少目前确实没找到过这种做法
    # 但怎么判断gold文档呢？数据集的外部知识库是怎么构造的？？？或者说外部知识库中的每条文本和数据集这里给出的每条文本是什么关系？？？
    # title的意义是什么？回答问题/检索文本与title有关吗？
        [
            {
                "idx": 0, 
                "title": "Adam Collis", 
                "paragraph_text": "Adam Collis is an American filmmaker and actor. He attended the Duke University from 1986 to 1990 and the University of California, Los Angeles from 2007 to 2010. He also studied cinema at the University of Southern California from 1991 to 1997. Collis first work was the assistant director for the Scott Derrickson's short \"Love in the Ruins\" (1995). In 1998, he played \"Crankshaft\" in Eric Koyanagi's \"Hundred Percent\".", "is_supporting": false}, 
            {"idx": 1, "title": "Ed Wood", "paragraph_text": "Edward Davis Wood Jr. (October 10, 1924 \u2013 December 10, 1978) was an American filmmaker, actor, writer, producer, and director.", "is_supporting": true}, 
            {"idx": 2, "title": "Woodson, Arkansas", "paragraph_text": "Woodson is a census-designated place (CDP) in Pulaski County, Arkansas, in the United States. Its population was 403 at the 2010 census. It is part of the Little Rock\u2013North Little Rock\u2013Conway Metropolitan Statistical Area. Woodson and its accompanying Woodson Lake and Wood Hollow are the namesake for Ed Wood Sr., a prominent plantation owner, trader, and businessman at the turn of the 20th century. Woodson is adjacent to the Wood Plantation, the largest of the plantations own by Ed Wood Sr.", "is_supporting": false}, 
            {"idx": 3, "title": "Conrad Brooks", "paragraph_text": "Conrad Brooks (born Conrad Biedrzycki on January 3, 1931 in Baltimore, Maryland) is an American actor. He moved to Hollywood, California in 1948 to pursue a career in acting. He got his start in movies appearing in Ed Wood films such as \"Plan 9 from Outer Space\", \"Glen or Glenda\", and \"Jail Bait.\" He took a break from acting during the 1960s and 1970s but due to the ongoing interest in the films of Ed Wood, he reemerged in the 1980s and has become a prolific actor. He also has since gone on to write, produce and direct several films.", "is_supporting": false}, 
            {"idx": 4, "title": "Doctor Strange (2016 film)", "paragraph_text": "Doctor Strange is a 2016 American superhero film based on the Marvel Comics character of the same name, produced by Marvel Studios and distributed by Walt Disney Studios Motion Pictures. It is the fourteenth film of the Marvel Cinematic Universe (MCU). The film was directed by Scott Derrickson, who wrote it with Jon Spaihts and C. Robert Cargill, and stars Benedict Cumberbatch as Stephen Strange, along with Chiwetel Ejiofor, Rachel McAdams, Benedict Wong, Michael Stuhlbarg, Benjamin Bratt, Scott Adkins, Mads Mikkelsen, and Tilda Swinton. In \"Doctor Strange\", surgeon Strange learns the mystic arts after a career-ending car accident.", "is_supporting": false}, 
            {"idx": 5, "title": "Tyler Bates", "paragraph_text": "Tyler Bates (born June 5, 1965) is an American musician, music producer, and composer for films, television, and video games. Much of his work is in the action and horror film genres, with films like \"Dawn of the Dead, 300, Sucker Punch,\" and \"John Wick.\" He has collaborated with directors like Zack Snyder, Rob Zombie, Neil Marshall, William Friedkin, Scott Derrickson, and James Gunn. With Gunn, he has scored every one of the director's films; including \"Guardians of the Galaxy\", which became one of the highest grossing domestic movies of 2014, and its 2017 sequel. In addition, he is also the lead guitarist of the American rock band Marilyn Manson, and produced its albums \"The Pale Emperor\" and \"Heaven Upside Down\".", "is_supporting": false}, 
            {"idx": 6, "title": "Sinister (film)", "paragraph_text": "Sinister is a 2012 supernatural horror film directed by Scott Derrickson and written by Derrickson and C. Robert Cargill. It stars Ethan Hawke as fictional true-crime writer Ellison Oswalt who discovers a box of home movies in his attic that puts his family in danger.", "is_supporting": false}, 
            {"idx": 7, "title": "Scott Derrickson", "paragraph_text": "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer. He lives in Los Angeles, California. He is best known for directing horror films such as \"Sinister\", \"The Exorcism of Emily Rose\", and \"Deliver Us From Evil\", as well as the 2016 Marvel Cinematic Universe installment, \"Doctor Strange.\"", "is_supporting": true}, 
            {"idx": 8, "title": "Ed Wood (film)", "paragraph_text": "Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood. The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau. Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones, Lisa Marie, and Bill Murray are among the supporting cast.", "is_supporting": false}, 
            {"idx": 9, "title": "Deliver Us from Evil (2014 film)", "paragraph_text": "Deliver Us from Evil is a 2014 American supernatural horror film directed by Scott Derrickson and produced by Jerry Bruckheimer. The film is officially based on a 2001 non-fiction book entitled \"Beware the Night\" by Ralph Sarchie and Lisa Collier Cool, and its marketing campaign highlighted that it was \"inspired by actual accounts\". The film stars Eric Bana, \u00c9dgar Ram\u00edrez, Sean Harris, Olivia Munn, and Joel McHale in the main roles and was released on July 2, 2014.", "is_supporting": false}
        ]
    }


{
    "dataset": "hotpotqa", 
    "question_id": "5a7fb1765542992e7d278d20", 
    "question_text": "Why does the City of Boston get their christmas tree from Ainslee Glen Nova Scotia?", 
    "level": "hard", 
    "type": "bridge", 
    "answers_objects": 
        [
            {
                "number": "", 
                "date": {"day": "", "month": "", "year": ""}, 
                "spans": ["for their assistance after the 1917 Halifax Explosion"]
            }
        ], 
    "contexts": 
        [
            {"idx": 0, "title": "Christmas tree production in Canada", "paragraph_text": "Christmas tree production in Canada totals from 3 to 6 million trees annually. Trees are produced in many of the provinces of Canada but the nation's leading producers are found in Quebec, Nova Scotia and Ontario, which account for 80 percent of Canadian tree production. Of the 900,000 trees produced annually in British Columbia, most are cut from native pine stands.", "is_supporting": false}, 
            {"idx": 1, "title": "Ainslie Glen, Nova Scotia", "paragraph_text": "Ainslie Glen (Scottish Gaelic: \"Gleann nam M\u00e0gan\" ) is a small community in the Canadian province of Nova Scotia, located in Inverness County on Cape Breton Island. In 2016, a tree on Crown Lands was chosen to become the Boston Christmas Tree.", "is_supporting": true}, 
            {"idx": 2, "title": "Boston Christmas Tree", "paragraph_text": "The Boston Christmas Tree is the City of Boston, Massachusetts' official Christmas tree. A tree has been lit each year since 1941, and since 1971 it has been given to the people of Boston by the people of Nova Scotia in thanks for their assistance after the 1917 Halifax Explosion. The tree is lit in the Boston Common throughout the Christmas season.", "is_supporting": true}, 
            {"idx": 3, "title": "Benjamin Green (merchant)", "paragraph_text": "Benjamin Green (July 1, 1713 \u2013 October 14, 1772) was a merchant, judge and political figure in Nova Scotia. He served as administrator for Nova Scotia in 1766 and from 1771 to 1772. He was born in Salem Village (later Danvers, Massachusetts), the son of the Reverend Joseph Green and Elizabeth Gerrish, and entered business with his brothers in Boston. In 1737, he married Margaret Pierce. He was secretary to William Pepperrell, who led the attack against Louisbourg in 1745, and served as treasurer for the forces from New England and secretary for the council that administered Louisbourg after its capture. In 1749, he went to Halifax, where he was named to Edward Cornwallis's Nova Scotia Council and also served as naval officer. Green was also judge in the vice admiralty court; he resigned in 1753. In 1750, he became secretary to the Council and provincial treasurer. Green was named a justice of the peace in 1760. While in England to assist in auditing the accounts of Peregrine Thomas Hopson, he had to defend himself against charges of assigning contracts to Malachy Salter in exchange for a share in the profits. He was reprimanded but allowed to retain his posts. During his term as administrator in 1766, he was criticized by the provincial assembly for not following the correct procedures for dealing with the provincial finances. Green resigned his post as provincial treasurer in 1768, citing poor health.", "is_supporting": false}, 
            {"idx": 4, "title": "French Village, Nova Scotia", "paragraph_text": "French Village is a rural community of the Halifax Regional Municipality in the Canadian province of Nova Scotia on Chebucto Peninsula. French village initially included present day villages of Tantallon, Glen Haven and French Village. The French that migrated to the area were French speaking families from the Principality of Montbeliard (annexed by France 1793)and known as the \"Foreign Protestants\". They had come to Nova Scotia between 1750 and 1752 to settle Lunenburg, Nova Scotia. Contrary to belief, they were not Huguenots. In 1901, the Halifax and Southwestern Railway was built through the area and the railway choose the name French Village for the station serving the three communities. The French Village station, actually located in Tantallon, has been preserved as a cafe beside the recreational trail that follows the old Halifax & Southwestern Railway roadbed.", "is_supporting": false}, 
            {"idx": 5, "title": "Christmas Island, Nova Scotia", "paragraph_text": "Christmas Island, Nova Scotia \"(Scottish Gaelic: Eilean na Nollaig)\" is a Canadian community of the Cape Breton Regional Municipality in Cape Breton, Nova Scotia. It has a post office, a firehall and a very small population. It also has a beach with access to the Bras d'Or lakes, and a pond that runs into the lake. Christmas Island got its name because of a native that lived there whose surname was Christmas. He died on Ghost Island, adjacent to the beach. The original inhabitants of the land, the Mi'kmaw people, called the area \"Abadakwich\u00e9ch\", which means \"the small reserved portion.\"", "is_supporting": false}, 
            {"idx": 6, "title": "Malachy Salter", "paragraph_text": "Malachy Salter (February 28, 1715 \u2013 January 13, 1781), a Nova Scotia merchant and office-holder, was born at Boston, second son of Malachy Salter and Sarah Holmes. He married Susanna Mulberry, on 26 July 1744 in Boston, and they had at least 11 children. He died at Halifax, Nova Scotia and is buried in the Old Burying Ground (Halifax, Nova Scotia) (His son Malachi Salter (d.1752) has the oldest grave marker in the burying ground).", "is_supporting": false}, 
            {"idx": 7, "title": "Chicago Christmas Tree", "paragraph_text": "The first official Christmas tree in the city of Chicago was installed in 1913 in Grant Park and lit on Christmas Eve by then-mayor Carter Harrison. This first tree was a 35 ft tall spruce tree. In December 1956 the official tree, though still installed in Grant Park (at Michigan Avenue and Congress Parkway), was not an individual tree. The tree was a combination of many smaller trees, stood 70 ft tall, and was decorated with over 4000 lights and 2000 ornaments. Beginning with Christmas 1966 the official Chicago Christmas tree was placed in Civic Center Plaza, now known as Daley Plaza. With the exception of 1981, the tree has been installed in Daley Plaza ever since.", "is_supporting": false}, 
            {"idx": 8, "title": "Mount Ingino Christmas Tree", "paragraph_text": "The Mount Ingino Christmas Tree is a lighting illumination in the shape of a Christmas tree that is installed annually on the slopes of Mount Ingino (Monte Ingino in Italian) outside the city of Gubbio, in Umbria region in Italy. The tree is also called the Gubbio Christmas Tree or \"the biggest Christmas tree in the world\". In 1991 the Guinness Book of Records named it \"The World's Largest Christmas Tree\".", "is_supporting": false}, 
            {"idx": 9, "title": "Foster Hutchinson (jurist)", "paragraph_text": "Foster Hutchinson Jr. (d. 1815) was a member of the Nova Scotia Council and one of the Puisne judges of the Supreme Court of Nova Scotia. He was the only son of Foster Hutchinson (judge), Sr., the nephew of Governor of Massuchsetts Thomas Hutchinson and grandchild of Governor of Nova Scotia Paul Mascarene. He arrived in Halifax from Boston with his father as Loyalists (1776). Hutchinson became a lawyer and worked under Chief Justice Thomas Andrew Lumisden Strange. Sir George Pr\u00e9vost appointed him an Assistant Justice to the Supreme Court (1809). He is buried in the Old Burying Ground (Halifax, Nova Scotia).", "is_supporting": false}]
}