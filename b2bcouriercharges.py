import pandas as pd
import plotly.graph_objects as go

#change directories

order_report = pd.read_csv("C:/Users/tanma/Downloads/b2b/Order Report.csv")
sku_master = pd.read_csv("C:/Users/tanma/Downloads/b2b/SKU Master.csv")
pincode_mapping = pd.read_csv("C:/Users/tanma/Downloads/b2b/pincodes.csv")
courier_invoice = pd.read_csv("C:/Users/tanma/Downloads/b2b/Invoice.csv")
courier_company_rates = pd.read_csv("C:/Users/tanma/Downloads/b2b/Courier Company - Rates.csv")

#checking the data 
print("Order Report:")
print(order_report.head())
print("\nSKU Master:")
print(sku_master.head())
print("\nPincode Mapping:")
print(pincode_mapping.head())
print("\nCourier Invoice:")
print(courier_invoice.head())
print("\nCourier Company rates:")
print(courier_company_rates.head())

#scanning for missing values 
print("\nMissing values in Website Order Report:")
print(order_report.isnull().sum())
print("\nMissing values in SKU Master:")
print(sku_master.isnull().sum())
print("\nMissing values in Pincode Mapping:")
print(pincode_mapping.isnull().sum())
print("\nMissing values in Courier Invoice:")
print(courier_invoice.isnull().sum())
print("\nMissing values in courier company rates:")
print(courier_company_rates.isnull().sum())

#there are random unnamed columns which also incidentally hold a lot of null values. so it is better to just drop them all together at the start itself.

order_report = order_report.drop(columns=['Unnamed: 3', 'Unnamed: 4'])
sku_master = sku_master.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
pincode_mapping = pincode_mapping.drop(columns=['Unnamed: 3', 'Unnamed: 4'])

#merging the Order Report and SKU Master (thru SKU)
merged_data = pd.merge(order_report, sku_master, on='SKU')
print(merged_data.head())

#renaming ExternOrderNo column to Order ID (other DF has Order ID)
merged_data = merged_data.rename(columns={'ExternOrderNo': 'Order ID'})


#We first extract the unique customer pin codes from the pincode mapping dataset (abc_courier)
#We then select specific columns (3 mentioned below) from the courier_invoice dataset (courier_abc)
#merge the ‘courier_abc’ DataFrame with the ‘abc_courier’ DataFrame based on the ‘Customer Pincode’ column. This merge operation helps us associate customer pin codes with their respective orders and shipping types (pincodes)

abc_courier = pincode_mapping.drop_duplicates(subset=['Customer Pincode'])
courier_abc= courier_invoice[['Order ID', 'Customer Pincode','Type of Shipment']]
pincodes= courier_abc.merge(abc_courier,on='Customer Pincode')
print(pincodes.head())


merged2 = merged_data.merge(pincodes, on='Order ID')
merged2['Weights (Kgs)'] = merged2['Weight (g)'] / 1000



#The weight_slab() function is defined to determine the weight slab based on the weight of the shipment. 
#The function first calculates the remainder of the weight divided by 1 and rounds it to one decimal place. If the remainder is 0.0, it means the weight is a multiple of 1 KG, and the function returns the weight as it is.
#If the remainder is greater than 0.5, it means that the weight exceeds the next half KG slab
#If the remainder is less than or equal to 0.5, it means the weight falls into the current half-KG bracket
#basically, this function is used as a rounding off function

def weight_slab(weight):
    i = round(weight % 1, 1)
    if i == 0.0:
        return weight
    elif i > 0.5:
        return int(weight) + 1.0
    else:
        return int(weight) + 0.5

merged2['Weight Slab (KG)'] = merged2['Weights (Kgs)'].apply(weight_slab)
courier_invoice['Weight Slab Charged by Courier Company']=(courier_invoice['Charged Weight']).apply(weight_slab)

#renaming columns
courier_invoice = courier_invoice.rename(columns={'Zone': 'Delivery Zone Charged by Courier Company'})
merged2 = merged2.rename(columns={'Zone': 'Delivery Zone As Per ABC'})
merged2 = merged2.rename(columns={'Weight Slab (KG)': 'Weight Slab As Per ABC'})


#we loop through each row of the ‘merged2’ DataFrame to calculate the expected charges based on ABC’s tariffs. 
#We retrieve the necessary rates and parameters(fixed charges and surcharges per weight tier), based on the delivery area
#Next we determine the weight of the slab for each row. If the shipment type is ‘Forward Charges’, we calculate the additional weight beyond the basic weight slab (0.5 KG) and apply the corresponding additional charges.
#For “Forward and RTO Charges” shipments, we consider additional charges for term and RTO components
#we store the calculated expected charges in the “Expected charges according to ABC” column of the “merged2” DataFrame

expected_charge = []

for _, row in merged2.iterrows():
    fwd_category = 'fwd_' + row['Delivery Zone As Per ABC']
    fwd_fixed = courier_company_rates.at[0, fwd_category + '_fixed']
    fwd_additional = courier_company_rates.at[0, fwd_category + '_additional']
    rto_category = 'rto_' + row['Delivery Zone As Per ABC']
    rto_fixed = courier_company_rates.at[0, rto_category + '_fixed']
    rto_additional = courier_company_rates.at[0, rto_category + '_additional']

    weight_slab = row['Weight Slab As Per ABC']

    if row['Type of Shipment'] == 'Forward charges':
        additional_weight = max(0, (weight_slab - 0.5) / 0.5)
        expected_charge.append(fwd_fixed + additional_weight * fwd_additional)
    elif row['Type of Shipment'] == 'Forward and RTO charges':
        additional_weight = max(0, (weight_slab - 0.5) / 0.5)
        expected_charge.append(fwd_fixed + additional_weight * (fwd_additional + rto_additional))
    else:
        expected_charge.append(0)


merged2['Expected Charge as per ABC'] = expected_charge
print(merged2.head())

#final merged output 

merged_output = merged2.merge(courier_invoice, on='Order ID')
print(merged_output.head())

#Calculating the difference in charges and the Expected Charges

df_diff = merged_output
df_diff['Difference (Rs.)'] = df_diff['Billing Amount (Rs.)'] - df_diff['Expected Charge as per ABC']
df_new = df_diff[['Order ID', 'Difference (Rs.)', 'Expected Charge as per ABC']]
print(df_new.head())


#Last step is to summarize the accuracy of B2B courier charges based on the charged prices and the expected prices

# Calculating the total orders for each category
total_correctly_charged = len(df_new[df_new['Difference (Rs.)'] == 0])
total_overcharged = len(df_new[df_new['Difference (Rs.)'] > 0])
total_undercharged = len(df_new[df_new['Difference (Rs.)'] < 0])

# Calculating the total amount for each category
amount_overcharged = abs(df_new[df_new['Difference (Rs.)'] > 0]['Difference (Rs.)'].sum())
amount_undercharged = df_new[df_new['Difference (Rs.)'] < 0]['Difference (Rs.)'].sum()
amount_correctly_charged = df_new[df_new['Difference (Rs.)'] == 0]['Expected Charge as per ABC'].sum()

# Creating a summary df
summary_data = {'Description': ['Total Orders where ABC has been correctly charged','Total Orders where ABC has been overcharged','Total Orders where ABC has been undercharged'],
                'Count': [total_correctly_charged, total_overcharged, total_undercharged],
                'Amount (Rs.)': [amount_correctly_charged, amount_overcharged, amount_undercharged]}

df_summary = pd.DataFrame(summary_data)
print(df_summary)

#the below code is just to add a bit of visualisation. completely optional.


fig = go.Figure(data=go.Pie(labels=df_summary['Description'], values=df_summary['Count'], textinfo='label+percent', hole=0.4))
fig.update_layout(title='Proportion')
fig.show()
