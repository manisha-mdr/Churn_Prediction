from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import sklearn
#import matplotlib
from sklearn.preprocessing import StandardScaler

#create a flask app
app = Flask(__name__)
#load pickle file
ML = pickle.load(open('ML.pkl','rb'))

@app.route('/')
def Home():
    return render_template('index.html')

#standard_to = StandardScaler()

@app.route('/predict',methods=['POST'])
def predict():
        
   gender = float(request.form['Gender'])
   Age = int(request.form['Age'])
   days_since_last_login = int(request.form['days_since_last_login'])
   avg_time_spent = int(request.form['avg_time_spent'])
   avg_transaction_value = float(request.form['avg_transaction_value'])
   points_in_wallet = float(request.form['points_in_wallet'])
   region_category = int(request.form['RegionCategory'])
   membership_category = float(request.form['Membership Category'])
   year = float(request.form['Year'])
   joined_through_Referral= float(request.form['Joined through Referral'])
   preferred_offer_type = float(request.form['Credit/Debit Card Offer'])
   Medium_of_Operation = float(request.form['Destop'])
   Internet_Option = float(request.form['Fiber Optics'])
   Used_Special_Discount=  float(request.form['Yes'])
   Offer_Application_Preference=  float(request.form['Offer Application Preference'])
   Past_Complain_Code=  float(request.form['Past Complain Code'])
   Complaint_Status =  float(request.form['Complaint Status'])
   Feedback=  float(request.form['Feedback'])

   # mean 

   mean = np.array([3.71579407e+01, 1.47780429e+01, 2.44074003e+02, 2.93369300e+04,
       6.87842642e+02, 5.02789166e-01, 7.45513811e-01, 2.25428456e+00,
       4.99143088e+00, 1.27683312e+00, 9.93514349e-01, 1.98928019e+00,
       1.00047046e+00, 5.48188722e-01, 5.52624504e-01, 4.99932791e-01,
       1.62967941e+00, 3.07947443e+00])

   std = np.array([1.58493603e+01, 1.00266614e+01, 3.99781181e+02, 1.95352767e+04,
       1.90203793e+02, 5.03207953e-01, 6.98566891e-01, 1.73810155e+00,
       8.19580569e-01, 7.03066377e-01, 8.14210857e-01, 1.02737548e+00,
       8.17305462e-01, 4.97672429e-01, 4.97222950e-01, 4.99999995e-01,
       1.22007244e+00, 2.56447895e+00])

   features = np.array([Age,days_since_last_login,
                                 avg_time_spent,
                                 avg_transaction_value,
                                 points_in_wallet,
                                 gender,
                                 region_category, 
                                 membership_category,
                                 year,
                                 joined_through_Referral,
                                 preferred_offer_type,
                                 Medium_of_Operation,
                                 Internet_Option,
                                 Used_Special_Discount,
                                 Offer_Application_Preference,
                                 Past_Complain_Code,
                                 Complaint_Status,
                                 Feedback])

   scaled_data = (features-mean)/std
   scaled_data = scaled_data.reshape(1,18)
   print(scaled_data.shape)
   prediction = ML.predict(scaled_data)
   print(prediction)
   
   if (prediction < 3):   
      return render_template('index.html', prediction_text ='The Customer will churn')
   else:
      return render_template ('index.html', prediction_text ='The Customer will not churn ')
 
if __name__=="__main__":
    app.run(debug= True)
        