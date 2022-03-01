import pandas as pd

csv_list = []
ans_list = [str(i) for i in range(18)]
### ensemble csv 파일 경로 지정
csv_list.append(pd.read_csv('./output/effnetb03_best_acc_ensemble.csv'))
csv_list.append(pd.read_csv('./output/effnetb03_best_acc_ensemble_2.csv'))
csv_list.append(pd.read_csv('./output/effnetb03_best_acc_ensemble_3.csv'))

for df in csv_list[1:]:
    csv_list[0][ans_list] += df[ans_list]

csv_list[0]['ans'] = csv_list[0][ans_list].idxmax(axis = 1).astype(int)
# csv_list[0]['ans'] = csv_list[0]['ans']
csv_list[0][['ImageID', 'ans']].to_csv('./output/' + 'ensemble.csv', index = False)