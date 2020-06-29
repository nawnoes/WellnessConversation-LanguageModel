import openpyxl
from openpyxl import Workbook, load_workbook

def tweet_dialog_dataset():
  root_path = "../data"
  tweet_file = root_path + "/tweeter_dialog_dataset.xlsx"
  tweet_file_output = root_path + "/tweeter_dialog_data.txt"

  f = open(tweet_file_output, 'w')

  wb = load_workbook(filename=tweet_file)

  ws = wb[wb.sheetnames[0]]
  # print(sheet)
  for row in ws.iter_rows():
    for cell in row:
      if cell.value == None:
        break
      # print(cell.value)
      f.write(cell.value + "\n")
    # print("\n\n\n")
    f.write("\n\n\n")

  f.close()

def wellness_question_data():
  root_path = "../data"
  wellness_file = root_path + "/wellness_dialog_dataset.xlsx"
  wellness_q_output = root_path + "/wellness_dialog_question.txt"

  f = open(wellness_q_output, 'w')

  wb = load_workbook(filename=wellness_file)

  ws = wb[wb.sheetnames[0]]
  # print(sheet)
  for row in ws.iter_rows():
    f.write(row[0].value + "    " + row[1].value + "\n")

  f.close()

def wellness_answer_data():
  root_path = "../data"
  wellness_file = root_path + "/wellness_dialog_dataset.xlsx"
  wellness_a_output = root_path + "/wellness_dialog_answer.txt"

  f = open(wellness_a_output, 'w')
  wb = load_workbook(filename=wellness_file)
  ws = wb[wb.sheetnames[0]]

  for row in ws.iter_rows():
    if row[2].value == None:
      continue
    else:
      f.write(row[0].value + "    " + row[2].value + "\n")
  f.close()

def wellness_category_data():
  root_path = "../data"
  wellness_file = root_path + "/wellness_dialog_dataset.xlsx"
  wellness_c_output = root_path + "/wellness_dialog_category.txt"

  f = open(wellness_c_output, 'w')

  wb = load_workbook(filename=wellness_file)

  ws = wb[wb.sheetnames[0]]
  # print(sheet)
  pre_category = ''
  category_count = 0
  flag = True
  for row in ws.iter_rows():
    if flag:
      flag = False
      continue
    if pre_category != row[0].value:
      f.write(row[0].value + "    " + str(category_count) + "\n")
      pre_category = row[0].value
      category_count += 1

  f.close()

def wellness_text_classification_data():
  root_path = "../data"
  wellness_category_file = root_path + "/wellness_dialog_category.txt"
  wellness_question_file = root_path + "/wellness_dialog_question.txt"
  wellness_text_classification_file = root_path + "/wellness_dialog_for_text_classification.txt"

  cate_file = open(wellness_category_file, 'r')
  ques_file = open(wellness_question_file, 'r')
  text_classfi_file = open(wellness_text_classification_file, 'w')

  category_lines = cate_file.readlines()
  cate_dict = {}
  for line_num, line_data in enumerate(category_lines):
    data = line_data.split('    ')
    cate_dict[data[0]] = data[1][:-1]
  print(cate_dict)

  ques_lines = ques_file.readlines()
  ques_dict = {}
  for line_num, line_data in enumerate(ques_lines):
    data = line_data.split('    ')
    # print(data[1]+ "    " + cate_dict[data[0]])
    text_classfi_file.write(data[1][:-1] + "    " + cate_dict[data[0]] + "\n")

  cate_file.close()
  ques_file.close()
  text_classfi_file.close()

def wellness_dialog_for_autoregressive():
  None

if __name__ == "__main__":
  root_path = "../data"
  wellness_file = root_path + "/wellness_dialog_dataset.xlsx"
  wellness_answer_file = root_path+"/wellness_dialog_answer.txt"
  wellness_question_file = root_path+"/wellness_dialog_question.txt"
  wellness_autoregressive_file = root_path+"/wellness_dialog_for_autoregressive.txt"

  answ_file = open(wellness_answer_file, 'r')
  ques_file = open(wellness_question_file, 'r')
  autoregressive_file = open(wellness_autoregressive_file, 'w')

  answ_lines = answ_file.readlines()
  ques_lines = ques_file.readlines()
  ques_dict = {}
  for line_num, line_data in enumerate(ques_lines):
    ques_data = line_data.split('    ')
    for ans_line_num, ans_line_data in enumerate(answ_lines):
      ans_data = ans_line_data.split('    ')
      if ques_data[0] == ans_data[0]:
        autoregressive_file.write(ques_data[1][:-1]+"    "+ans_data[1])
      else:
        continue




  cate_file.close()
  ques_file.close()
  text_classfi_file.close()