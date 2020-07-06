
import pandas as pd 
import numpy as np
import copy, sys

from preprocessing_mimic2_lib.FeatureUtils import clean6, clean4

class ICUEventSet():

    def __init__(self, totalNumPatients, rareFrequency, rareFrequencyCategory):
        self.allEventNames = set()                  # set of strings
        self.allEventValuesWithDup = dict()
        self.allEventCategories = dict()
        self.totalNumPatients =totalNumPatients
        self.rareFrequency = rareFrequency
        self.summary = (0,0,0,0,0)
        self.indicatorTypes = dict()
        self.categoricalTypes = dict()
        self.rareFrequencyCategory = rareFrequencyCategory
        self.allNaEvents = dict()
        self.allNaTypes = dict()

    def GetSummary(self):
        '''
        Returns current composition of seen events.

        Returns:
            5-tuple of ints:
            (# Total features, # Indicators, # Categorical,
             # Continuous, # All NA)

        '''
        countIdxs = {'indicator': 1, 'categorical': 2,
                     'continuous': 3, 'all_na': 4}
        counts = [0] * 5

        for event in self.allEventCategories:
            eventCategory = self.allEventCategories[event]
            counts[countIdxs[eventCategory]] += 1

        counts[0] = sum(counts)
        self.summary = tuple(counts)
        return self.summary

    def Contains(self, event):
        return event in self.allEventNames

    def AddEvent(self, eventName, eventValuesWithDup, allNa):
        '''
        Add event to seen events and classify by their observed values,
        store their categories as well.

        Args:
            eventName: str
            eventValuesWithDup: list of ? type
            allNa: bool, whether all observed values were originally na

        Modifies:
            self.allEventNames: set of strs
            self.allEventCategories: dictionary with keys of string type

        '''
        self.allEventNames.add(eventName)
        # self.allEventValuesWithDup[eventName] = eventValuesWithDup

        if allNa:
            self.allEventCategories[eventName] = 'all_na'
            self.allNaEvents[eventName] = len(eventValuesWithDup)
        else:
            self.allEventCategories[eventName] = \
                    self._CategorizeEvent(eventName, eventValuesWithDup)
            # if eventName == 'lab:blood:chemistry:creatine_kinase,_mb_isoenzyme':
            #     print(eventValuesWithDup)
            #     assert(False)

    def RegisterEventEndingWithColons(self, eventName):
        eventRoot = eventName[:-1] # exclude the ending colon
        self.eventsEndingWithColons[eventName] = eventRoot


    def _CategorizeEvent(self, eventName, eventValuesWithDup):
        '''
        Categorize events.

        Updates:
            - Now indicator values could tolerate nan.
            - [Ongoing] Using static mappings instead of dynamic processing to determine
            event category.

        '''

        if eventName.startswith('prescribed'):
            return 'continuous'

        # 1. screen for indicators 
        indicatorValueSet = {'not_done', 'done', 'yes', 'no', '1', '0', '1.0', '0.0'}
        eventValuesWithDupInd = set(RemoveNaInList(eventValuesWithDup, remove = ['na', 'na*', 'other']))
        eventValuesWithDupInd = [str(ev) for ev in eventValuesWithDupInd]
        indicatorCondition = set(eventValuesWithDupInd).issubset(indicatorValueSet)

        if indicatorCondition:
            manuallyExcluded = {'chart:calprevflg:kg', 'chart:weight_change:kg'}
            if eventName not in manuallyExcluded:
            # print(eventName, eventValuesWithDup)
            # print("----------------------------")
                return 'indicator'  

        categoricalSpecialCases = \
            {'chart:no:',
             'chart:d-dimer_(0-500):',
             'lab:urine:chemistry:length_of_urine_collection',
             'chart:rsbi_deferred:',
             'chart:rsbi_not_completed:',
             'chart:pupil_size_right:',
             'chart:pupil_size_left:',
             'chart:pupil_size_left:na',
             'chart:pupil_size_right:na',
             'chart:ct_#1_suction_amount:',
             'chart:edema_amount:',
             'chart:ct_#2_suction_amount:'}

        continuousSpecialCases = \
            {'lab:urine:hematology:renal_epithelial_cells',
             'lab:urine:hematology:transitional_epithelial_cells',
             'lab:blood:chemistry:troponin_i',
             'lab:blood:hematology:cd19',
             'lab:blood:hematology:cd20',
             'lab:blood:hematology:cd5',
             'lab:other_body_fluid:hematology:cd19',
             'lab:other_body_fluid:hematology:cd3',
             'chart:i:e_ratio:na',
             'lab:blood:chemistry:human_chorionic_gonadotropin',
             'lab:urine:chemistry:bicarbonate,_urine',
             'lab:urine:hematology:wbc_casts',
             'lab:blood:hematology:d-dimer'}

        indicatorSpecialCases = \
            {'chart:parameters_checked:na',
             'chart:back_care:na',
             'chart:skin_care:na'}

        if eventName in categoricalSpecialCases:
            self.categoricalTypes[eventName] = self._CategorizeCategorical(eventName, \
                                                                   eventValuesWithDup)
            return 'categorical'
        elif eventName in continuousSpecialCases:
            return 'continuous'
        elif eventName in indicatorSpecialCases:
            return 'indicator'

        # 3. 'categorical' | 'mixed' | 'continuous'
        origianlEventValuesWithDup = copy.deepcopy(eventValuesWithDup)
        eventValuesWithDup, floatCount, totalCount = \
                ScreenForContinuous(eventName, eventValuesWithDup)
        if floatCount/totalCount <= 0.02:
            self.categoricalTypes[eventName] = self._CategorizeCategorical(\
                                        eventName, origianlEventValuesWithDup)
            return 'categorical'
        elif floatCount/totalCount >= 0.3:
            return 'continuous'
        else:
            if 'neg' in eventValuesWithDup:
                if eventName in ['lab:urine:hematology:bacteria', 
                                 'lab:blood:chemistry:acetone']:
                    self.categoricalTypes[eventName] = \
                            self._CategorizeCategorical(eventName, \
                                                    origianlEventValuesWithDup)
                    return 'categorical' 

                else:
                    # these are 'lab:urine:hematology:xxx' events
                    # the 'neg' values will be converted to -1 in later steps
                    return 'continuous'                

        print('=======================================')
        print('[ALERT] This event cannot be classified.')
        print(floatCount/totalCount, floatCount/self.totalNumPatients)
        print(eventName, set(eventValuesWithDup), len(eventValuesWithDup))
        print(eventValuesWithDup)
        print('=======================================')
        print('\n')

        return None

    def _CategorizeIndicator(self, eventName, eventValuesWithDup):
        '''
        Categorize indicator events into:
            - Single value: 'done', '1' etc.
            - Double value: 'yes', 'no', 'done' etc.

        Args:
            - 

        Modifies:
            self.indicatorTypes: key: str, mapto: 1 or 2

        '''
        return # not used currently 

        eventValuesWithoutDup = set(eventValuesWithDup)
        if len(eventValuesWithoutDup) == 1:
            # single value for sure
            return 2

        positiveSet = {'yes', 'done', 1}
        negativeSet = {'no', 0}
        eventValuesWithoutDupMerged = set()
        for eventValue in eventValuesWithoutDup:
            if eventValue in positiveSet:
                eventValuesWithoutDupMerged.add('+')
            elif eventValue in negativeSet:
                eventValuesWithoutDupMerged.add('-')
            else:
                raise NotImplementedError

    def IndicatorType(self, indicatorEventName):
        '''
        Args:
            indicatorEventName

        Returns:
            indicatorType: this has been classified during 'understandEvents'

        '''
        return self.indicatorTypes[indicatorEventName]

    def _CategorizeCategorical(self, eventName, eventValuesWithDup):
        '''
        This is a prep step for one hot encoding. Substeps include:
            - Recognize possible categories.
            - Merge replicate categories.
            - Merge infrequent categories.
            - If categories are leveled, encode by integer values rather than
              one-hot encode.
            - If categories are actually continuous data that somehow got 
              misclassified, corrected here.

        Modifies:
            ?
        ------
        - events that need special handling
            - leveled
            - continuous
            - dups
        - events that do not need special handling
            - rare events/categories
            - rare categories that are not meaningful could be dropped

        concerns:
            na values
            single out leveled categorical
            merge rare
        ------
        Besides events with special handling instructions returned here
        all other events are one-hot encoded on the fly


        # REWRITE DOC FOR THIS FUNCTION !

        '''

        # First, all event values are cleaned by the same standards that will
        # be used when actually loading events to patients:
        #           eventValue = clean6(eventName, rawEventValue)

        eventValuesWithDup = [clean6(eventName, eventValue) \
                                           for eventValue in eventValuesWithDup]

        # Then, special cases are handled case-by-case
        # Currently, the first two edge cases do not support null handling :(

        if 'lab:blood:hematology' in eventName:
            levels = ['normal', 'occasional', '1+', '2+', '3+']    
            # this class of events are all leveled categorical
            if eventName == 'lab:blood:hematology:platelet_smear':
                # na: unable .. due to platelet clotting 
                levels = ['rare', 'very_low', 'low', 'normal', \
                                                    'high', 'very_high', 'na']
            elif eventName == 'lab:blood:hematology:d-dimer':
                levels = ['<500', '500-1000', '1000-2000', '>2000']

            observedSet = set(eventValuesWithDup)
            levelSet = set(levels)
            # check if intersect
            if levelSet & observedSet:
                return levels 

        elif 'lab:urine:hematology' in eventName:
            labUrineHematologyMapping = {'none': 0, 'neg': 0, '0': 0,
                                         'rare': 1, 'few': 1, 'sm': 1, 'tr': 1,
                                         'f': 1,
                                         'occ': 2, 'mod': 2, 'o': 2, 'm': 2,
                                         'many': 3, 'lg': 3, 'lge': 3}
            if set(eventValuesWithDup).issubset(\
                                        set(labUrineHematologyMapping.keys())):
                # 0 = none, neg
                # 1 = rare, few, sm, tr
                # 2 = occ, mod
                # 3 = many, lg, lge
                return labUrineHematologyMapping

        elif eventName in ['chart:ett_size_(id):na', 'chart:ett_mark_(cm):na',\
                           'chart:ett_mark:na', 'chart:airway_size:na',\
                           'chart:i:e_ratio:na', 'chart:i:e_ratio:']:
            # continuous misclassified 
            return 'continuous'          # drop down to continuous handling

        elif eventName in ['chart:verbal_response:na', 'chart:motor_response:na',\
                          'chart:eye_opening:na', 'chart:pain_level_(rest):na',\
                          'chart:pain_level/response:na']:
            # leveled categorical
            return 'continuous'

        elif eventName in ['chart:no:']:
            return

        # rare categories are combined
        return self._MergeRareCategories(eventName, eventValuesWithDup)

    def _MergeRareCategories(self, eventName, eventValuesWithDup):
        '''
        The input events are categorical events to be one-hot encoded.
        However, certain categories contain too few instacnes and require
        merging.

        Args: -

        Returns:
            mergeInstructions: set of to-be-merged categories
            None: no need to merge, simply one-hot encode

        '''
        categoriesIn = list(set(eventValuesWithDup))
        categoriesOut = [x for x in categoriesIn]
        categoriesCount = [eventValuesWithDup.count(x) for x in categoriesIn]

        categoriesIdxs = sorted(list(range(len(categoriesIn))), \
                                key=categoriesCount.__getitem__)

        categoriesCount.sort()

        cumSum = 0
        mergeIdx = 0
        rareThreshold = self.rareFrequencyCategory * self.totalNumPatients

        while mergeIdx < len(categoriesCount) and \
            (cumSum < rareThreshold or \
            categoriesCount[mergeIdx] < rareThreshold):

            cumSum += categoriesCount[mergeIdx]
            mergeIdx += 1

        mergedCategoryIdxs = categoriesIdxs[:mergeIdx]
        mergedCategories = [categoriesIn[i] for i in mergedCategoryIdxs]

        if len(mergedCategories) != 1:
            return set(mergedCategories)

    def CategoricalType(self, categoricalEventName):
        return self.categoricalTypes[categoricalEventName]

    def AnalyzeAllNaEvents(self):
        '''
        All-na events fall under one of two categories: events with timestamps,
        and events that have already been one-hot encoded. This function looks
        at observed all-na events and return handling instructions.

        Modifies:
            self.allNaTypes: dict() of handling instructions
                - 'timestamp'
                - 'indicator'
                - 'drop'
                - set()
                - None

        '''
        allNaEvents = sorted(self.allNaEvents.keys())
        # These are known base categories of all-na events
        knownCategories = {'marital_status:', 'insurance:',\
                           'gender:', 'ethnicity:', 'admission_type:', \
                           'admission_location:', 'discharge_diagnosis:', \
                           'microbiology:blood_culture:', 'microbiology:csf;spinal_fluid:',\
                           'microbiology:mrsa_screen:', 'microbiology:catheter_tip-iv:', \
                           'microbiology:sputum:', 'microbiology:stool:', 'microbiology:swab:',\
                           'microbiology:urine:', 'microbiology:bile:', 'microbiology:fluid,other:',\
                           'microbiology:foreign_body:','microbiology:peritoneal_fluid:',\
                           'microbiology:tissue:', 'microbiology:abscess:', 'microbiology:aspirate:',\
                           'microbiology:biopsy:', 'microbiology:blood_culture_-_neonate:',\
                           'microbiology:blood_culture_(_myco/f_lytic_bottle):',\
                           'microbiology:bronchial_washings:', 'microbiology:bronchoalveolar_lavage:',\
                           'microbiology:dialysis_fluid:', 'microbiology:ear:', 'microbiology:eye:', \
                           'microbiology:fluid_received_in_blood_culture_bottles:',\
                           'microbiology:fluid_wound:', 'microbiology:foot_culture:',\
                           'microbiology:influenza_a/b_by_dfa:', 'microbiology:isolate:',\
                           'microbiology:joint_fluid:', 'microbiology:mini-bal:',\
                           'microbiology:peritoneal_fluid:', 'microbiology:pleural_fluid:',\
                           'microbiology:rapid_respiratory_viral_screen_&_culture:',\
                           'microbiology:staph_aureus_screen:','microbiology:throat_culture:',\
                           'microbiology:throat_for_strep:', 'microbiology:viral_culture:r/o_herpes_simplex_virus:'}

        currBaseCategory = None
        currBaseCategoricalValues = []

        for event in allNaEvents:
            if ':' not in event:        # these are definitely not categorical
                self.allNaTypes[event] = 'timestamps'
            else:
                unrecognized = True
                for knownCategory in knownCategories:
                    if event.startswith(knownCategory):
                        unrecognized = False

                        eventName = ':'.join(event.split(':')[:-1])
                        if eventName != knownCategory[:-1]:
                            print("[WARNING] Fatal error occurr while handling all-na events.")
                            print(event, eventName, knownCategory)
                            assert(False)

                        if currBaseCategory == None or \
                            currBaseCategory != eventName:

                            if len(currBaseCategoricalValues) != 0:
                                self.allNaTypes[currBaseCategory] = \
                                    self._MergeRareCategories(eventName, \
                                                    currBaseCategoricalValues)

                            currBaseCategory = eventName
                            currBaseCategoricalValues = []

                        numOccur = self.allNaEvents[event]
                        currBaseCategoricalValues.extend([event]*numOccur)

                if unrecognized:
                    if self.allNaEvents[event] >= \
                                    self.totalNumPatients * self.rareFrequency:
                        # remove rare all-na events
                        # the remaining events are all na-flag for continuous events
                        print("[WARNING] Unrecognized all-na events %s, not one-hot encoded" % event)
                        self.allNaTypes[event] = 'indicator'
                    else:
                        self.allNaTypes[event] = 'drop'

        if len(currBaseCategoricalValues) != 0: # a last empty action required
            self.allNaTypes[eventName] = \
                self._MergeRareCategories(currBaseCategory, \
                                                    currBaseCategoricalValues)

    def AllNaType(self, allNaEventName):
        if allNaEventName in self.allNaTypes:
            eventKey = allNaEventName
        else:
            eventKey = ':'.join(allNaEventName.split(':')[:-1])

        if eventKey not in self.allNaTypes:
            print('[Debug] Unrecognized all-na events {}'.format(allNaEventName))
            # assert(False)
        
        return self.allNaTypes[eventKey]

    def MakeCopy(self):
        '''
        A lite copy returned for lookup purposes.

        '''
        returnedCopy = ICUEventSet(totalNumPatients=self.totalNumPatients,\
                                   rareFrequency=self.rareFrequency,\
                                   rareFrequencyCategory=self.rareFrequencyCategory)
        returnedCopy.allEventNames = copy.deepcopy(self.allEventNames)
        # returnedCopy.allEventValuesWithDup = copy.deepcopy(self.allEventValuesWithDup)
        returnedCopy.allEventCategories = copy.deepcopy(self.allEventCategories)
        # returnedCopy.summary = copy.deepcopy(self.summary)
        returnedCopy.indicatorTypes = copy.deepcopy(self.indicatorTypes)
        returnedCopy.categoricalTypes = copy.deepcopy(self.categoricalTypes)
        # returnedCopy.allNaEvents = copy.deepcopy(self.allNaEvents)
        returnedCopy.allNaTypes = copy.deepcopy(self.allNaTypes)
        # returnedCopy.eventsEndingWithColons = copy.deepcopy(self.eventsEndingWithColons)

        return returnedCopy

def ScreenForContinuous(eventName, eventValuesWithDup):
    '''
    This function loops through all observerations and screens for potentially
    continuous event using an arbitrary heuristic -- ratio of 'floatable' values.

    Args:
        - 

    Returns:
        floatCount: the number of observations that can be turned to float
        totalCount: the number of meaningful, non-na observations

    '''
    naSet = {'na', 'na*'}
    eventValuesWithDupNew = []
    floatCount = 0
    totalCount = 0

    for value in eventValuesWithDup:

        try:
            floatValue = float(clean4(value))    # clean units
            floatCount += 1 
            totalCount += 1
        except:
            if value not in naSet:
                totalCount += 1
            floatValue = value

        eventValuesWithDupNew.append(floatValue)

    return eventValuesWithDupNew, floatCount, totalCount

def RemoveNaInList(l, remove = ['na', 'na*']):
    result = []
    naSet = set(remove)
    for x in l:
        if x not in naSet:
            result.append(x)
    return result

























