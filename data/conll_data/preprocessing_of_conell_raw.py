
#####to remove ,, ;, :
# # infile = "test_raw_conll.txt"
# # outfile = "test1.txt"
# #
delete_list = ["-", ":", ",", '"', "...", "(", ")", "'s", "''", "'"]
delete_list1 = ["-DOCSTART-","-X-","-X-","O"]

# # fin = open(infile)
# # fout = open(outfile, "w+")
# # for line in fin:
# #     #for word in delete_list:
# #         #line = line.replace(word, "") used to replace the particualr matched word with empty space.
#
#
#     fout.write(line)
# fin.close()
# fout.close()



######to remove empty rows

# import re
# with open("pos1.txt", "r") as f:
#     lines = f.readlines()
# with open("pos2.txt", "w") as f:
#     for line in lines:
#         if line.strip("\n") != ' ':
#             f.write(line)



#####get the line in one line using . as the next line
# with open("pos2.txt", "r+") as f:
#     data = f.readlines()
#     for line1 in data:
#         if line1.strip().split(" ")[0] != '.' and line1.strip().split(" ")[0] != 'POS':
#             #if line1.strip().split(" ")[1] is not None:
#                 print((line1.strip().split(" ")[0]),end= " " )
#             #
#         else:
#             print()




#####To remove the commas lines and also the doc start lines


with open("validationraw_conll.txt", "r") as f:
    lines = f.readlines()
with open("valid1.txt", "w") as f:
    x = []
    for line in lines:
        x = line.strip("\n").split(" ")
        if x[0] not in delete_list:
            if x[0] not in delete_list1:
                f.write(line)
        else:
            print()



####check if all the lines size is 4 and if not delete the line


with open("valid1.txt", "r") as f:
    lines = f.readlines()
with open("validation.txt", "w") as f:
    x = []
    for line in lines:
        x = line.strip("\n").split(" ")
        if len(x)==4:
                f.write(line)
        else:
            print()