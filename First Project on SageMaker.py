#!/usr/bin/env python
# coding: utf-8

# In[28]:


get_ipython().run_line_magic('conda', 'install -c conda-forge shap')


# In[29]:


import shap


# In[30]:


X, y = shap.datasets.adult()


# In[31]:


X_diasplay, y_display = shap.datasets.adult(display=True)


# In[32]:


feature_names= list(X.columns)


# In[33]:


feature_names


# In[34]:


display(X.describe())


# In[35]:


gist = X.hist(bins=30, sharey=True, figsize=(20,10))


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


X_train, X_val, y_trian, y_val = train_test_split(X_train, y_train, test_size=-0.25, random_state=1)

X_train_display = X_display.loc(X_train.index)
# In[ ]:


import pandas as pd


# In[ ]:


train = pd.concat([pd.Series(y_train, index=X_train.index, name="Income>50K", dtype=int), X_train], axis=1)


# In[ ]:


validation = pd.concat([pd.Series(y_val, index=X_val.index, name="Income>50K", dtype=int), X_val], axis=1)


# In[ ]:


test=pd.concat([pd.Series(y_test, index=X_test.index, name="Income>50K", dtype=int), X_test], axis=1)

trainvalidation
# In[ ]:


test


# In[ ]:


train.to_csv('train.csv', index=False, header=False)

validation.to_csv('validation.csv', index=False, header=False)import sagemaker, boto3, os
# In[ ]:


bucket =sagemkaer.Session().default_bucket()


# prefix="demo-sagemaker-xgboost-adult-income-prediction"

# In[ ]:


boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'data/train.csv')).upload_file('train.csv')

boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'data/validation.csv')).upload_file('validation.csv')
# In[ ]:


get_ipython().system(' aws s3 is (buvket)/(prefix)/data --recursive')


# In[ ]:


import sagemaker

region = sagemaker.Session().boto_region_nameprint("AWS Region: {}".format(region))
# In[ ]:


role = sagemaker.get_execution_role()


# In[ ]:


print("RoleArn: {}".format(role))


# In[ ]:


sagemaker.__version__


# In[ ]:


from sagemaker.debugger import Rule, rule_configs


# In[ ]:


from sagemaker.session import TrainingINput


# In[ ]:


s3_output_location = 's3://{}/{}/{}'.format(bucket, prefix, 'xgboost_model')


# In[ ]:


container = sagemaker.image_urls.retrieve("xgboost", region, "1.2-1")


# In[ ]:


xgm_model =sagameker.estimator.Estimator(image_uri=container, role=role, instance_count=1, instance_type='ml.m4.xlarge', volume_size=5, output_path=s3_output_location, sagemaker_session=sagemaker.Session(), rule=[Rule.sagemaker(rule_configs.create_xgboost_report())])


# In[ ]:


xgb_model.set_gyperparameters(max_depth=5, eta=0.2, gamma=4, min_child_weight=6, subsample=0.7, objective="binary:logistic", num_round=1000)


# In[ ]:


from sagamaker.session import TrainingInput


# In[ ]:


train_input = TrainingInput("s3://{}/{}/{}".format(bucket, prefix, "data/train.csv", content_type="csv"))


# In[ ]:


validation_input = TrainingInput("s3://{}/{}/{}".format(bucket, prefix, "data/validation.csv"), content_type="csv")


# In[ ]:


xhb_model.fit("train:", train_input, "validation": validation_input), wait=True


# In[ ]:


rule_output_path = xgb_model.output_path + "/" + xgb_model.latest_training_job.name + "/rule-output"


# In[ ]:


get_ipython().system(' aws s3 ls {tule_output_path}  -- recursive')


# In[ ]:


from IPython.display import FileLink, FileLinks


# In[ ]:


display('Click Linke below to view the XGBoost Training report', FileLink("CreateXgboostReport/zgboost_report.html"))


# In[ ]:


profile_report_name = (rule["RuleConfigurationName"]
                      for rule in xgb_model.latest_training_job.rule_job_summary()
                      if "Profiler" in rule["RuleConfigurationName"]][0]
                      profiler_report_name
                      display("Click link below to view profiel report", FileLink(profiler_report_name=+"/profieler-output/profiler-report.html")))


# In[ ]:


xgb_model.model_data


# In[ ]:


import sagemaker


# In[ ]:


from sagemaker.serializers import CSVSerializer


# In[38]:


xgb_predictor = xgb_model.deploy(
initial_instance_count=1, 
instance_type='ml.t2.medium', 
serializaer=CSVSerializer())


# In[39]:


xgb_predictor.endpoint_name


# In[40]:


import numpy as np


# In[44]:


def predict(data, rows=1000):
    split_array = np.array_split(data, int(data.shape[0]/float(rows)+1))
    predictions = ''
    for array in split_array"
    predictions = ','.join([predictions, xgb_predictor.predict(array).decode('utf-8')])
    return np.fronstring(predictions[1:], sep=',')


# In[45]:


import matplotlib.pyplot as plt


# In[46]:


predictions = predict(test.to_numpy()[:,1:])


# In[47]:


plt.hist(predictions)


# In[48]:


plt.show()


# In[49]:


import sklearn


# In[50]:


cutoff = 0.5


# In[51]:


print(sklearn.metrics.confusion_matrix(test.iloc[:, 0], np.where(predictions>cutoff, 1, 0))


# In[53]:


print(sklearn.metrics.classification_report(test_iloc[:, 0], np.where(predictions>cutoff, 1, 0)))


# In[54]:


import matplotlib.pyplot as plt


# In[55]:


cutoffs = np.aranger(0.01, 1, 0.01)


# In[56]:


log_loss = []


# In[57]:


for c in cutoffs:
    log_loss.append(
    sklearn.metrics.log_loss(test.iloc[:, 0], np.where(predictions>c, 1, 0)))


# In[58]:


plt.figure(figsize=(15, 10))


# In[59]:


plt.plot(cutoffs, log_loss)


# In[60]:


plt.xlabel("Cutoff")


# In[61]:


plt.ylabel("Log loss")


# In[62]:


plt.show()


# In[64]:


print('Log loss is minimized at a cutoff of', cutoffs[np.argmin(log_loss)], 
     '.and the log loss value at the minimum is', npp.min(log_loss)


# In[ ]:




