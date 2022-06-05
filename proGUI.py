import PySimpleGUI as sg
import comment_extract as CE
import sentimentYouTube as SYT

sg.theme('DarkBlue1')	# Add a touch of color
# All the stuff inside your window.
layout = [  [sg.Text('                                    ')],
            [sg.Text('Video Link',size=(18,1) , justification='left',font=('Franklin Gothic Book', 18, 'bold')),sg.InputText( justification='center')],
            [sg.Text('                                    ')],
            [sg.Text('Comments Number',size=(18,1) , justification='left',font=('Franklin Gothic Book', 18, 'bold')), sg.InputText(justification='center')],
            [sg.Text('                                    ')],
            [sg.Text(size=(32,1)),sg.Button('Comments Analysis',size=(18,1) ,font=('Franklin Gothic Book', 12, 'bold'))],
            [sg.Text('                                    ')],
            [sg.Text('Positive Comments',size=(18,1) , justification='left', text_color='green',font=('Franklin Gothic Book', 18, 'bold')),
             sg.Text('0%', size=(10,1),justification='center', background_color='black', text_color='green', font=('Digital-7',14), key="_Positive_")],
            [sg.Text('                                    ')],
            [sg.Text('Negative Comments',size=(18,1) , justification='left', text_color='red', font=('Franklin Gothic Book', 18, 'bold')),
             sg.Text('0%', size=(10,1),justification='center', background_color='black', text_color='red', font=('Digital-7',14), key="_Negative_")],
            [sg.Text('                                    ')],
            [sg.Text(' ', size=(50,1),justification='center', background_color='white', text_color='black', font=('Digital-7',16), key="_out_")],
            [sg.Text('                                    ')]]
            

# Create the Window
window = sg.Window('Window Title', layout)    


# STEP3 - the event loop
while True:
    event, values = window.read()   # Read the event that happened and the values dictionary
    if event in (None, 'Exit'):     # If user closeddow with X or if user clicked "Exit" button then exit
        break
    if event == 'Comments Analysis':
        if values[0]!='' :
            c=0
            if (values[1]=='') or (not 0 < float(values[1])) :
                c=10
            else:
                c=float(values[1])
            videoLink= values[0]
            videoId = videoLink.split("=", 1)[1]
            print(videoId)
            comments = CE.commentExtract(videoId,c)
            print(comments)
            pos,nev=SYT.sentiment(comments,videoId)
            window['_Positive_'].update(value=pos+'%')
            window['_Negative_'].update(value=nev+'%')
            if float(pos)>float(nev):
                if float(pos)>75 :
                    window['_out_'].update(value='Video content and feedback are "EXCELLENT"')
                else :
                    window['_out_'].update(value='Video content and feedback are "GOOD"')
            else:
                if float(nev)>75 :
                    window['_out_'].update(value='Video content and feedback are "BAD"')
                else :
                    window['_out_'].update(value='Video content and feedback are "Very BAD"')
        else:
            sg.popup('YOU Must Enter Video URL' )

window.close()
   

