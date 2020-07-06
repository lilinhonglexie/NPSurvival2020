
import re, copy
import numpy as np
from scipy import stats

def clean1(eventName):
    '''
    Replace space with _, merge known duplicates.

    Args:
        eventName: raw reading from EHR

    Returns:
        eventName: cleaned version

    '''
    eventName = eventName.replace(' ', '_')
    return eventName

def clean2(eventValuesWithDup):
    '''
    Replace all values that are equivalent to 'na' (but not originally NAN) 
    with 'na*'. 

    Args:
        eventValuesWithDup: raw readings from EHR, np.array of string

    Returns:
        eventValuesWithDup: cleaned up version
        len(eventValuesWithDup): number of patients with this event
        allNa: whether the observed list only contains na values

    '''

    # commonly seen values that are equiv to 'na'
    # we change these to na* to indicate that they were originally not na
    # and that we do not want to one-hot encode them
    naSet = {"", "error", "notdone", "**_info_not_available_**",
              'unable_t_report', "unable_to_report",
              'error,unable_to_report', "discarded",
              'computer_network_failure._test_not_resulted.',
              "error_specimen_clotted", "specimen_clotted",
              'disregard_result', "test_not_done", 'unable_to_determine',
              'unable_to_determine:', 'unable_to_repoet',
              'unable_to_repert', 'not_reported', 'specimen_clottted',
              "unable_to_repot", 'spec._clotted',
              'spec.clotted', 'unable_to_analyze', 'no_data', 
              '.', '..', '...', '....', '.....', 'unabl', 'unable', 
              'specimen_contaminated', 'spec_clotted'}
    naPattern = r".*(disregard_previous|error_previously" + \
                r"_reported|disregard_result).*"

    eventValuesWithDupNew = []
    warningFlag = False

    for eventValue in eventValuesWithDup:
        if type(eventValue) == float and np.isnan(eventValue):
            eventValueNew = "na"
            warningFlag = True

        elif type(eventValue) == str:
            eventValueNew = eventValue.replace(" ", "_")
            if eventValueNew in naSet or \
                re.match(r'unable_to.*due_to.*', eventValueNew) or \
                re.match(naPattern, eventValueNew):
                # print('[Debug] Found NAN* values.{}'.format(eventValueNew))
                eventValueNew = "na*"

        else:
            # print("[TESTING] New incoming type found:")
            # print(eventValue, type(eventValue))
            eventValueNew = str(eventValue)
                
        eventValuesWithDupNew.append(eventValueNew)

    # if 'unknown' in eventValuesWithDupNew: # unknown
    #     print(set(eventValuesWithDup), set(eventValuesWithDupNew))

    return eventValuesWithDupNew, len(eventValuesWithDupNew), \
           set(eventValuesWithDupNew) == {'na'}, warningFlag

def clean3(eventValue):
    '''
    Args:
        eventValue: continuous event value, 
                    potentially having neg, na values etc.

    Returns:
        eventValue: clean up neg and na values

    '''
    eventValueNew_, _, _, _ = clean2([eventValue])
    eventValueNew = eventValueNew_[0]

    # continuous na set
    continuousNaSet = {'na', 'na*'}
    if eventValueNew in continuousNaSet:
        return 'na'
    elif eventValueNew in ['neg', 'negative']:
        return 0
    return eventValueNew

def clean4(eventValue):
    '''
    Args:
        eventValue: continuous event value,
                    potentially having ranges, units

    Returns:
        eventValue: convert ranges to numbers
                    Note: this is under the condition that most of the observed
                    values under this event are already numbers, 
                    rather than ranges.

    '''
    if type(eventValue) != str:
        return eventValue

    pattern3 = r'^(?P<value>([0-9]*[.])?[0-9]+)_?' + \
                r'(cm|mm|grams|gmmag|mgso4|gm_mag|gmg+|o1b|mg|cs|kcl|meq|units?|fs|min|ng/ml)?$'
    pattern1 = r'^(?P<low>[+-]?([0-9]*[.])?[0-9]+)_?-_?(?P<hi>[+-]?([0-9]*[.])?[0-9]+)$' 
    pattern2 = r'[a-z_:]*(<|>|<=|>=|less_than|greater_than' + \
                r'|greater_thn|less_thn)_?(?P<value>([0-9]*[.])?[0-9]+)_?' + \
                r'(cm|mm|grams|gmmag|mgso4|gm_mag|gmg+|o1b|mg|cs|kcl|meq|units?|fs|min|ng/ml)?$'


    # pattern1 = r'.*([0-9]+)_?-_?([0-9]+).*'
    # pattern2 = r'[a-z_:]*(<|>|<=|>=|less_than|greater_than' + \
    #             r'|greater_thn|less_thn)_?([0-9]+).*'
    # pattern3 = r'([0-9]+)_?(cm|mm|grams|gmmag|mgso4|gm_mag|gmg+|o1b)'
    pattern3Units = {'cm', 'mm', 'grams', 'gmmag', 'mgso4', 'gm_mag', 'gmg', 
                     'o1b', 'mg', 'cs', 'kcl', 'meq', 'units', 'unit', 'fs', 'min'}

    if re.match(pattern3, eventValue):
        # "n1 cm" => n1
        result = re.match(pattern3, eventValue)
        return float(result["value"])

    if re.match(pattern1, eventValue):
        # "n1-n2" => (n1+n2)/2
        result = re.match(pattern1, eventValue)
        return (float(result["low"]) + float(result["hi"])) / 2

    if re.match(pattern2, eventValue):
        # ">n1" | "<=n1" | ... => n1
        result = re.match(pattern2, eventValue)
        return float(result["value"])

    for unit in pattern3Units:
        # remaining cases for hard-to-get-rid of units
        if eventValue.endswith(unit) or eventValue.startswith(unit):
            eventValue = eventValue.replace(unit, '')
            return eventValue.replace('_', '')  
    
    return eventValue

def clean5(eventName, eventValue):
    '''
    Hardcoded edge cases when handling continuous values

    Args:
        eventName: - 
        eventValue: continuous event value, potentially strings

    Returns:
        eventValue: after considering hardcoded cases

    '''
    if type(eventValue) != str:
        return eventValue

    if eventName == 'chart:glucose_(70-105):na':
        return eventValue.replace('cs', '')                 #99cs --> 99

    if eventName.startswith("prescribed") and eventName.endswith("unit"):
        if type(eventValue) == str:
            return eventValue.replace(',', '')            # 25,000 --> 25000

    if eventName == 'lab:blood:hematology:d-dimer':
        try:
            return float(eventValue)
        except:
            return 'na'

    if eventName == 'lab:blood:hematology:ptt':
        if eventValue == '150_is_highest_measured_ptt':
            return 150

    if 'chart:pain_level' in eventName or \
        eventName in ['chart:verbal_response:na', 'chart:motor_response:na',\
                      'chart:eye_opening:na']:

        if 'unable_to_score' in eventValue:
            return 'na'

        firstChar = eventValue[0]
        
        if firstChar.isdigit():
            secondChart = eventValue[1]
            if secondChart.isdigit():
                return int(firstChar+secondChart)
            return int(firstChar)

        # commonly seen cases
        eventValue = eventValue.replace('.', '')
        eventValue = eventValue.replace('-', '_')

        painlevelMapping = {'none': 0,
                            'none_to_mild': 1,
                            'mild': 2,
                            'mild_to_mod': 3, 'mild_to_moderate': 3,
                            'moderate': 5,
                            'mod_to_severe': 6, 'moderate_to_severe': 6,
                            'severe': 8,
                            'severe-worst': 9,
                            'worst': 10}

        if eventValue in painlevelMapping:
            return painlevelMapping[eventValue]
        else:
            print('[WARNING] Unrecognized continuous value for chart:pain_level, etc.')
            print(eventName, eventValue)
            return 'na'

    if eventName in ['chart:i:e_ratio:', 'chart:i:e_ratio:na']:
        # attempt 1:
        try:
            left, right = eventValue.split(':')
            return 1/float(right)
        except:
            pass

        # attemp 2:
        try:
            right = float(eventValue)
            return 1/float(right)
        except:
            return 'na'

    return eventValue

def clean6(eventName, eventValue):
    '''
    Clean up categorical values. Remove space; replace known na values with na;
    merge known duplicates that might cause issues during one-hot encoding.

    Args:
        - 

    Returns:
        eventValue: cleaned up

    '''
    eventValueNew_, _, _, _ = clean2([eventValue])
    eventValueNew = eventValueNew_[0]

    # continuous na set
    categoricalNaSet = {'na', 'na*'}
    if eventValueNew in categoricalNaSet:
        return 'na'

    dupValueRef = {
                "chart:marital_status": {'w': 'widowed', 'm': 'married',
                                         's': 'single', 'd': 'divored'},
                "chart:marital_status:na": {'w': 'widowed', 'm': 'married',
                                            's': 'single', 'd': 'divored'},
                "lab:urine:hematology:urine_appearance": {"slcloudy": "slcldy"},
                "lab:urine:hematology:urine_color": {'y': 'yellow',
                                                     'yel': 'yellow', 's': 'straw',
                                                     'drkamber': 'dkamber',
                                                     'dkambe': 'dkamber',
                                                     'dkamb': 'dkamber',
                                                     'amb': 'amber'},
                'lab:urine:hematology:renal_epithelial_cells': {2.0: '0-2'},
                'lab:urine:hematology:transitional_epithelial_cells':\
                                                                  {1.0: '0-2'},
                'chart:anti-embolism_device:na':\
                                     {'compress_sleeves': 'compression_sleeve'},
                'chart:daily_wake_up:na': {'no,_not_sedated': 'no/not_sedated'}}

    if eventName in dupValueRef:
        if eventValueNew in dupValueRef[eventName]:
            return dupValueRef[eventName][eventValueNew]

    elif 'chart:pupil_size' in eventName:
        return eventValueNew.replace('_', '')

    # elif eventName == 'lab:blood:hematology:d-dimer':

    #     try:
    #         eventValueNew = float(eventValueNew)

    #         if eventValueNew < 500: 
    #             eventValueNew = '<500'
    #         elif eventValueNew >= 500 and eventValueNew < 1000: 
    #             eventValueNew = '500-1000'
    #         elif eventValueNew >= 1000 and eventValueNew < 2000:
    #             eventValueNew = '1000-2000'
    #         elif eventValueNew >= 2000:  
    #             eventValueNew = '>2000'
    #         else:
    #             print('[Debug] Unrecognized float value for lab:blood:hematology:d-dimer')

    #     except:
    #         if eventValueNew == '1000-20': eventValueNew = '1000-2000'
    #         if eventValueNew not in ['<500', '500-1000', '1000-2000', '>2000']:
    #             print('[Debug] Unrecognized str value for lab:blood:hematology:d-dimer')

    elif eventName == 'chart:no:':
        if eventValueNew not in ['40ppm', '41ppm', 'off']:
            eventValueNew = 'non-encoded-numeric'
        else:
            pass

    return eventValueNew

def parseBloodGasVentilationRate(ventilationRate):
    '''
    Args:
        ventilationRate: leftValue/rightValue

    Returns:
        leftValue, rightValue

    '''
    left, right = ventilationRate.split('/')
    if left != '':
        leftValue = int(left)
    else:
        leftValue = None

    if right != '':
        rightValue = int(right)
    else:
        rightValue = None

    return leftValue, rightValue















