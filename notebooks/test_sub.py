id = 'abfmneahjp'
labels = {'abfmneahjp_p0': 1, "abfmneahjp_p1": 0, 'abfmneahjp_p2': 0, "meow_123": 3}
possible_values = list(map(lambda value: f"{id}_p{value}", range(10)))
labels_list = []
for id, label in labels.items():
	if id in possible_values:
		labels_list.append(label)
print(sum(labels_list))
