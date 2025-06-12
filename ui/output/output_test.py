from ui.output.output import append_to_excel


data_list=[("./test_img.jpg",
            '123가 1234','321나 4321')]
append_to_excel(data_list, './차량정보.xlsx')