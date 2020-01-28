import glob
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style


style.use('ggplot')
matplotlib.rcParams['font.family'] = "DejaVu Sans"
matplotlib.rc('axes',edgecolor='darkgrey')


files = glob.glob("./tmp/stats*.json")

def should_include(data):
    evaluation_cnf = data["cnf"]["evaluation"]
    if evaluation_cnf["predictor"]["class"] != "ModelPredictor":
        return False
    utility = evaluation_cnf["predictor"]["args"].get("utility", 0)
    return 0.5 <= utility <= 1.5


def get_stats(data):
    return dict(
        average_gas_price=data["stats"]["average_gas_price"],
        average_blocks_waited=data["stats"]["average_blocks_waited"],
        utility=data["cnf"]["evaluation"]["predictor"]["args"]["utility"]
    )

stats = []
for filename in files:
    with open(filename) as f:
        data = json.load(f)
        if should_include(data):
            stats.append(get_stats(data))

stats.sort(key=lambda x: x["utility"])

x = [v["utility"] for v in stats]
y_wait = [v["average_blocks_waited"] for v in stats]
y_price = [v["average_gas_price"] for v in stats]

fig = plt.figure(figsize=(4.8041, 3))
ax = plt.subplot()
l1, = ax.plot(x, y_wait, color="royalblue", label="Blocks to wait")
ax.set_ylabel("Number of blocks to wait")
ax.set_xlabel("Urgency parameter value")
ax2 = ax.twinx()
l2, = ax2.plot(x, y_price, color="tomato", label="Gas price", )
ax2.set_ylabel("Gas price")
plt.legend(handles=[l1, l2], loc='upper center', prop={"size": 8})
ax.grid(color='whitesmoke')
ax.set_facecolor('white')
plt.tight_layout()
plt.savefig("urgency-effect.pdf")
