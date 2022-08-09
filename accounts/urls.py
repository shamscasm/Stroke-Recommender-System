from django.urls import path
from . import views 
from django.contrib.auth import views as auth_views

urlpatterns = [
    # this is the url path for register and login page , which sends it to views.py wherein it is linked to the .html file
    #these are related to the user's account
    path('register/',views.registerPage, name="register"),
    path('login/',views.loginPage, name="login"),
    path('logout/',views.logoutUser, name="logout"),

    #these are the pages the user can go to 
    path('landingPage/', views.landingPage, name="landingPage"),
    path('contact/', views.contact, name="contact"),

    path('view_pdf', views.view_pdf, name="view_pdf"),
 


# this is the default path
    path('',views.home, name="home"),











#these are the password reset page
    path('reset_password/', auth_views.PasswordResetView.as_view() , name="reset_password"),
    path('reset_password_sent/',auth_views.PasswordResetDoneView.as_view() , name="reset_password_sent"),
    path('reset/<uidb64>/<token>', auth_views.PasswordResetConfirmView.as_view, name="reset/<uidb64>/<token>"),
    path('reset_password_complete/', auth_views.PasswordResetCompleteView.as_view , name="reset_password_complet"),
    
    ]



