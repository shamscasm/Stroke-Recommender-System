 
from django.shortcuts import render, redirect
 
from .forms import CreateUserForm
from django.contrib.auth import authenticate , login , logout
from django.contrib import messages
 
from .models import*
 

#pdf display imports 
from django.http import FileResponse
import io

from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter


from model.main import *




#Create your views here.

def registerPage(request):
    form = CreateUserForm()

    if request.method== "POST":
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            user = form.cleaned_data.get('username')
            messages.success(request, "Account was created for "+user)
            return redirect ('login')

    context = {'form':form}
    return render(request , 'accounts/register.html', context)



def loginPage(request):

    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username= username, password = password)

        if user is not None:
            login(request, user)

            return redirect('landingPage')
        else:
            messages.error(request,'Username or Password is incorrect ')

    context = {}
    return render(request , 'accounts/login.html', context)



def logoutUser(request):
    logout(request)
    return redirect ('home')


def home(request):

    context = {}
    return render(request , 'accounts/home.html', context)




def landingPage(request):
    

    context = {}   
    return render(request, 'accounts/landingPage.html', context)



def contact(request):

    if request.method == "POST":
        pid= request.POST.get('PID')
        age= int(request.POST.get('age'))
        gender= int(request.POST.get('gender'))
        
        mrs= int(request.POST.get('mrs'))
        systolic= int(request.POST.get('systolic'))
        distolic= int(request.POST.get('distolic'))
        glucose= int(request.POST.get('glucose'))
        
        smoking= int(request.POST.get('smoking'))
        bmi= int(request.POST.get('bmi'))
        cholestrol= int(request.POST.get('cholestrol'))
        tos= int(request.POST.get('tos'))
        
        inplis = [age,gender, mrs,systolic,distolic,glucose,smoking,bmi,cholestrol,tos]

        #testlis = [54,1,5,180,123,233,1,26,225,3 ]

        arr = np.array(inplis)

        sdf_train= get_shap_explanation_scores_df(arr)
        x = get_risk_level(arr) 
        chart= plot_SHAP_result(inplis)

        Featurelis=[(str(sdf_train['Feature'][0])),(str(sdf_train['Feature'][1])),(str(sdf_train['Feature'][2])) , (str(sdf_train['Feature'][3])), (str(sdf_train['Feature'][4]))]
     
 
        context = {'chart':chart ,
                    'x':x,
                    'pid':pid,
                    'features':Featurelis,
                    'age':age,
                    'gender':gender,
                    'mrs':mrs,
                    'sys': systolic,
                    'dis': distolic,
                    'glucose':glucose,
                    'smoking': smoking,
                    'bmi':bmi,
                    'chol': cholestrol,
                    'tos':tos} 

    return render(request, 'accounts/contact.html', context)




def view_pdf(request):

    buf = io.BytesIO()

    c= canvas.Canvas(buf,pagesize= letter , bottomup=0)

    textob = c.beginText()
    textob.setTextOrigin(inch,inch) 
    textob.setFont("Helvetica",14)

    
    lines=[
        
    ]


    for line in lines:
        textob.textLine(line)

    c.drawText(textob)
    c.showPage()
    c.save()
    buf.seek(0)

    return FileResponse(buf, as_attachment=True , filename = 'Recommender_results.pdf')




