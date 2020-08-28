import pandas as pd 
import numpy as np


class Clean_data(object):
    '''
    For ease of use: Call the following methods on a full dataset
    drop_null()
    to_date_time()
    add_date()
    add_hour()
    set_values()

    Due to size of class, utilize date time and add times methods in separate stages
    '''

    def __init__(self, full_df):
        self.full_df = full_df
    

    def drop_null(self):
        '''
        method called to drop unnamed column and single nan text row
        full_df updates in place 
        '''
        try:
            self.full_df.drop(1101124, axis = 0, inplace = True)
            self.full_df.drop('Unnamed: 0', axis = 1, inplace = True)
        except:
            print('Be certain this is the full dataset and try again')

        

    def to_date_time(self):
        self.full_df['created_at'] = [pd.to_datetime(dt) for dt in self.full_df['created_at']]
        print('If you want hours and days as separate columns, call add_hour/add_date methods')


    def add_date(self):
        self.full_df['created'] = [dt.date() for dt in self.full_df['created_at']]

    def add_hour(self):
        self.full_df['created'] = [dt.hour for dt in self.full_df['created_at']]

    def set_values(self):
        '''
        Calling this method changes 'rating' column to zeros and ones:
        zero:rejected, one: approved

        Also binarizes the parent_id column to avoid null values
        zero:not available, one: available
        '''
        #binary ratings
        self.full_df.rating = (self.full_df.rating == 'approved')*1

        #parent id...present or not
        self.full_df.parent_id = (~self.full_df.parent_id.isna())*1
        
if __name__ == '__main__':
    print('Due to the size of the available data, download json bucket files 1-5 to utilize clean data class!')
    print('SEE: "Save and Concate all files into csv" section in EDA notebook for help concatinating files')




