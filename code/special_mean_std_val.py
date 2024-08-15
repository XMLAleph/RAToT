import statistics

print('#'*10+'Retrieval f1-score'+'#'*10)
print("IRCoT")
print("hotpotqa")
vals = [0.24298169325283228, 0.24660501538304716]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))
print("2wikimqa")
vals = [0.31947861323333276, 0.3253579075602276]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))
print("musique")
vals = [0.2187416163710413, 0.20057945298465213, 0.19926571932104942]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))

print("Sc-IRCoT")
print("hotpotqa")
vals = [0.242124919479457, 0.24764538036328826]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))
print("2wikimqa")
vals = [0.3423525847199707, 0.329841664196878, 0.3315611486102703]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))
print("musique")
vals = [0.21600601555264198, 0.196683087602109, 0.20316581139943238]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))

print("RAGoT")
print("hotpotqa")
vals = [0.4178357956054626, 0.43665735293775515, 0.45134060921974267]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))
print("2wikimqa")
vals = [0.47052668461098945, 0.43131471423622814, 0.45371423474274897]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))
print("musique")
vals = [0.3077453651387009, 0.317756974817982, 0.32279732567866015]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))

print("IRCoT")
print("hotpotqa")
vals = [0.24298169325283228, 0.24660501538304716]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))
print("2wikimqa")
vals = [0.31947861323333276, 0.3253579075602276]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))
print("musique")
vals = [0.2187416163710413, 0.20057945298465213, 0.19926571932104942]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))

print("Sc-IRCoT")
print("hotpotqa")
vals = [0.242124919479457, 0.24764538036328826]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))
print("2wikimqa")
vals = [0.3423525847199707, 0.329841664196878, 0.3315611486102703]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))
print("musique")
vals = [0.21600601555264198, 0.196683087602109, 0.20316581139943238]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))

print("RAGoT")
print("hotpotqa")
vals = [0.4178357956054626, 0.43665735293775515, 0.45134060921974267]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))
print("2wikimqa")
vals = [0.47052668461098945, 0.43131471423622814, 0.45371423474274897]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))
print("musique")
vals = [0.3077453651387009, 0.317756974817982, 0.32279732567866015]
print(round(statistics.mean(vals),3))
print(round(statistics.stdev(vals),3))

print('#'*10+'Answer em/f1-score'+'#'*10)
print("IRCoT")
print("hotpotqa")
vals = [(0.4,0.559),(0.34,0.492),(0.4,0.559)]
vals_1 = [elem[0] for elem in vals]
vals_2 = [elem[1] for elem in vals]
print(round(statistics.mean(vals_1),3))
print(round(statistics.stdev(vals_1),3))
print(round(statistics.mean(vals_2),3))
print(round(statistics.stdev(vals_2),3))
print("2wikimqa")
vals = [(0.48,0.659),(0.35,0.49),(0.47,0.618)]
vals_1 = [elem[0] for elem in vals]
vals_2 = [elem[1] for elem in vals]
print(round(statistics.mean(vals_1),3))
print(round(statistics.stdev(vals_1),3))
print(round(statistics.mean(vals_2),3))
print(round(statistics.stdev(vals_2),3))
print("musique")
vals = [(0.14,0.242),(0.07,0.169),(0.08,0.199)]
vals_1 = [elem[0] for elem in vals]
vals_2 = [elem[1] for elem in vals]
print(round(statistics.mean(vals_1),3))
print(round(statistics.stdev(vals_1),3))
print(round(statistics.mean(vals_2),3))
print(round(statistics.stdev(vals_2),3))

print("Sc-IRCoT")
print("hotpotqa")
vals = [(0.45, 0.587),(0.35,0.505),(0.44,0.595)]
vals_1 = [elem[0] for elem in vals]
vals_2 = [elem[1] for elem in vals]
print(round(statistics.mean(vals_1),3))
print(round(statistics.stdev(vals_1),3))
print(round(statistics.mean(vals_2),3))
print(round(statistics.stdev(vals_2),3))
print("2wikimqa")
vals = [(0.52,0.681),(0.4,0.544),(0.51,0.662)]
vals_1 = [elem[0] for elem in vals]
vals_2 = [elem[1] for elem in vals]
print(round(statistics.mean(vals_1),3))
print(round(statistics.stdev(vals_1),3))
print(round(statistics.mean(vals_2),3))
print(round(statistics.stdev(vals_2),3))
print("musique")
vals = [(0.18,0.269),(0.11,0.244),(0.15,0.27)]
vals_1 = [elem[0] for elem in vals]
vals_2 = [elem[1] for elem in vals]
print(round(statistics.mean(vals_1),3))
print(round(statistics.stdev(vals_1),3))
print(round(statistics.mean(vals_2),3))
print(round(statistics.stdev(vals_2),3))

print("RAGoT")
print("hotpotqa")
vals = [(0.37,0.549),(0.4,0.56),(0.4,0.552)]
vals_1 = [elem[0] for elem in vals]
vals_2 = [elem[1] for elem in vals]
print(round(statistics.mean(vals_1),3))
print(round(statistics.stdev(vals_1),3))
print(round(statistics.mean(vals_2),3))
print(round(statistics.stdev(vals_2),3))
print("2wikimqa")
vals = [(0.43,0.551),(0.38,0.518),(0.49,0.631)]
vals_1 = [elem[0] for elem in vals]
vals_2 = [elem[1] for elem in vals]
print(round(statistics.mean(vals_1),3))
print(round(statistics.stdev(vals_1),3))
print(round(statistics.mean(vals_2),3))
print(round(statistics.stdev(vals_2),3))
print("musique")
vals = [(0.2,0.287),(0.13,0.261),(0.17,0.281)]
vals_1 = [elem[0] for elem in vals]
vals_2 = [elem[1] for elem in vals]
print(round(statistics.mean(vals_1),3))
print(round(statistics.stdev(vals_1),3))
print(round(statistics.mean(vals_2),3))
print(round(statistics.stdev(vals_2),3))