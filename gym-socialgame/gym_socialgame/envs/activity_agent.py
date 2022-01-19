import pandas as pd
import numpy as np
import json
import random
import string
import os
import argparse

from pandas.core.arrays import boolean

class Utilities:
	def special_min(*args):
		return min(i for i in args if i is not None)
		
	def series_add(base : np.ndarray, addition : np.ndarray, addition_displacement = 0):
		for relative_time_step_index, addition_value in enumerate(addition):
			absolute_time_step_index = relative_time_step_index + addition_displacement
			if absolute_time_step_index >= len(base):
				return
			else:
				base[absolute_time_step_index] += addition_value

class ArrayRange:
	def __init__(self, start_index = 0, length = 1):
		self._start_index = start_index
		self._length = length

	def start_index(self):
		return self._start_index
	
	def end_index(self):
		return self.start_index() + len(self)

	def __len__(self):
		return self._length

	def __contains__(self, index):
		return index >= self.start_index() and index < self.end_index()
	
	def __str__(self):
		return "Start Index: {start_index}, Length: {length}".format(start_index = self.start_index(), length = len(self))

class ActivityEnvironment:
	'''
	props:
		- activities
		- activitiy_consumers
		- consumed_activities   |   Dict from activity to time consumed
	'''
	def __init__(self, activities, activity_consumers, time_domain):
		self._activities = activities
		self._activity_consumers = activity_consumers
		self._time_domain = time_domain
  
	def execute(self, energy_prices: np.ndarray):
		energy_prices = energy_prices - np.median(energy_prices)
		for time_step_index in range(len(energy_prices)):
			for activity_consumer in self._activity_consumers:
				activity_consumer.execute_step(energy_prices, time_step_index)
	
	def aggregate_demand(self, time_range: ArrayRange = None):
		if time_range is None:
			time_range = self._time_domain
		demand_by_activity_consumer = {}
		for activity_consumer in self._activity_consumers:
			consumer_demand = activity_consumer.aggregate_demand(time_range)
			demand_by_activity_consumer[activity_consumer] = consumer_demand
		return demand_by_activity_consumer

	def build(source_file_name = None):
		if source_file_name is None:
			source_file_name = "gym-socialgame/gym_socialgame/envs/activity_environments/activity_env.json"
		return JsonActivityEnvironmentGenerator.generate_environment(source_file_name)
	
	def restore(self):
		for activity in self._activities:
			activity.restore()
		for activity_consumer in self._activity_consumers:
			activity_consumer.restore()

	def execute_aggregate(self, energy_prices, time_range: ArrayRange = None):
		if time_range is None:
			time_range = self._time_domain
		self.execute(energy_prices)
		result = self.aggregate_demand(time_range)
		return result

	def restore_execute_aggregate(self, energy_prices, time_range: ArrayRange = None):
		if time_range is None:
			time_range = self._time_domain
		self.restore()
		result = self.execute_aggregate(energy_prices, time_range)
		return result

	def build_execute_aggregate(energy_prices, source_file_name = "gym-socialgame/gym_socialgame/envs/activity_environments/activity_env.json"):
		new_env : ActivityEnvironment = ActivityEnvironment.build(source_file_name)
		time_range = new_env._time_domain
		result = new_env.aggregate_execute(energy_prices, time_range)
		return result

	def get_activity_consumers(self):
		return self._activity_consumers


class ActivityConsumer:
	'''
	props:
		- ?activity_foresights	| 	Max amount forwards that activity can be moved by price difference
		- activity_values       |   Dict from activitiy to Series of active values by time
		- activity_thresholds    |   Dict from activity to Series of threshold to consume that activity by time
		- demand_unit_price_factor | Dict from demand unit to Series of energy price factor by time <-- willingness to change usage because of price
									? dependent variables: 1) given time of consumption
									? store in demand unit or activity
		- demand_unit_quantity_factor | Dict from demand unit to Series of total energy consumed factor by time <-- willingness to change usage because of price
									? dependent variables: 1) given time of consumption
									? store in demand unit or activity
		- consumed_activities   |   Dict from activity to time consumed
	'''

	def __init__(self, name, activity_values = None, activity_thresholds = None, demand_unit_price_factor = None, demand_unit_quantity_factor = None):
		
		self.name = name

		if activity_values is None:
			if not hasattr(self, "_activity_values"):
				self._activity_values = {}
		else:
			self._activity_values = activity_values
		
		self._initial_activity_values = ActivityConsumer.copy_activity_values(self._activity_values)
		
		if activity_thresholds is None:
			if not hasattr(self, "_activity_thresholds"):
				self._activity_thresholds = {}
		else:
			self._activity_thresholds = activity_thresholds
		
		if demand_unit_price_factor is None:
			if not hasattr(self, "_demand_unit_price_factor"):
				self._demand_unit_price_factor = {}
		else:
			self._demand_unit_price_factor = demand_unit_price_factor
		
		if demand_unit_quantity_factor is None:
			if not hasattr(self, "_demand_unit_quantity_factor"):
				self._demand_unit_quantity_factor = {}
		else:
			self._demand_unit_quantity_factor = demand_unit_quantity_factor

		self._consumed_activities = {}

	def setup(self, activity_values = None, activity_thresholds = None, demand_unit_price_factor = None, demand_unit_quantity_factor = None):
		
		if activity_values is None:
			if not hasattr(self, "_activity_values"):
				self._activity_values = {}
		else:
			self._activity_values = activity_values

		self._initial_activity_values = ActivityConsumer.copy_activity_values(self._activity_values)

		if activity_thresholds is None:
			if not hasattr(self, "_activity_thresholds"):
				self._activity_thresholds = {}
		else:
			self._activity_thresholds = activity_thresholds
		
		if demand_unit_price_factor is None:
			if not hasattr(self, "_demand_unit_price_factor"):
				self._demand_unit_price_factor = {}
		else:
			self._demand_unit_price_factor = demand_unit_price_factor
		
		if demand_unit_quantity_factor is None:
			if not hasattr(self, "_demand_unit_quantity_factor"):
				self._demand_unit_quantity_factor = {}
		else:
			self._demand_unit_quantity_factor = demand_unit_quantity_factor

	def execute_step(self, energy_price_by_time, time_step):
		to_calculate_for_times = ArrayRange(start_index = time_step)
		for activity, active_value_by_time in self._activity_values.items():
			if (activity._consumed is False or activity._consumed is None):
				price_effect_at_time = activity.price_effect_by_time(energy_price_by_time, to_calculate_for_times, self)
				# normalized_price_effect = (price_effect[time_step] - price_effect.median())
				total_value_for_time = active_value_by_time[time_step] + price_effect_at_time
				threshold_for_time = self._activity_thresholds[activity][time_step]

				if total_value_for_time > threshold_for_time:
					self.consume(time_step, activity)

	def consume(self, time_step, activity):
		self._consumed_activities[activity] = time_step
		activity.consume(time_step)

	def aggregate_demand(self, time_range: ArrayRange):
		total_demand = np.zeros(len(time_range), dtype = np.float64)
		for activity, time_consumed in self._consumed_activities.items():
			activity_demand = activity.aggregate_demand(time_consumed, time_range, self)
			total_demand += activity_demand
			# Utilities.series_add(total_demand, activity_demand, time_consumed)

		return total_demand

	def restore(self):
		self._consumed_activities = {}
		self._activity_values = ActivityConsumer.copy_activity_values(self._initial_activity_values)

	def copy_activity_values(activity_values):
		new_dict = {}
		for activity, time_values in activity_values.items():
			new_dict[activity] = time_values.copy()
		return new_dict


class Activity:
	'''
	props:
		- demand_units
		- effect_vectors | a dict of ActivityConsumers to a dict of Activities to a series of effect values/functions affecting activity values by relative time
							? can also condition on the time consumed as effect may differ
		- consumed
	'''
	
	def __init__(self, name, demand_units = None, effect_vectors = None):
		self.name = name
		if demand_units is None:
			if not hasattr(self, "_demand_units"):
				self._demand_units = []
		else:
			self._demand_units = demand_units

		self.set_length()

		if effect_vectors is None:
			if not hasattr(self, "_effect_vectors"):
				self._effect_vectors = {}
		else:
			self._effect_vectors = effect_vectors
		self._consumed = False

	def setup(self, demand_units = None, effect_vectors = None):
		if demand_units is None:
			if not hasattr(self, "_demand_units"):
				self._demand_units = []
		else:
			self._demand_units = demand_units

		self.set_length()
	
		if effect_vectors is None:
			if not hasattr(self, "_effect_vectors"):
				self._effect_vectors = {}
		else:
			self._effect_vectors = effect_vectors

	def __len__(self):
		return self._length

	def set_length(self):
		if(self._demand_units is not None and len(self._demand_units) > 0):
			self._length = max(len(demand_unit) for demand_unit in self._demand_units)
		else:
			self._length = 0
			
	def price_effect_by_time(self, energy_price_by_time, time_range: ArrayRange, for_consumer: ActivityConsumer) -> pd.Series:
		if len(time_range) == 1:
			total = 0
			for demand_unit in self._demand_units:
				# adds price effect by time to total_price_effect
				total += demand_unit.price_effect_by_time(energy_price_by_time, time_range, for_consumer)
			return total
		else:
			total_price_effect = np.zeros(len(time_range), dtype = np.float64)
			for demand_unit in self._demand_units:
				# adds price effect by time to total_price_effect
				demand_unit.price_effect_by_time(energy_price_by_time, time_range, for_consumer, total_price_effect)
			return total_price_effect
	
	def consume(self, time_step):
		self._consumed = time_step
		for for_consumer, effect_vectors_by_activity in self._effect_vectors.items():
			for activity, effect_vector in effect_vectors_by_activity.items():
				active_values_by_time = for_consumer._activity_values[activity]
				Activity.effect_active_values(time_step, active_values_by_time, effect_vector)

	def consume(self, time_step):
		self._consumed = time_step
		for for_consumer, effect_vectors_by_activity in self._effect_vectors.items():
			for activity, effect_vector in effect_vectors_by_activity.items():
				active_values = for_consumer._activity_values[activity]
				Activity.effect_active_values(time_step, active_values, effect_vector)

	def effect_active_values(base_time_step_index, active_values, local_effect_vector):
		for relative_time_step_index, effect_value in enumerate(local_effect_vector):
			if effect_value is None or effect_value == 1:
				continue
			absolute_time_step_index = base_time_step_index + relative_time_step_index
			if absolute_time_step_index >= len(active_values):
				return
			else:
				active_values[absolute_time_step_index] *= effect_value

	def aggregate_demand(self, time_consumed, time_range: ArrayRange, for_consumer: ActivityConsumer):
		total_demand = np.zeros(len(time_range), dtype = np.float64)
		for demand_unit in self._demand_units:
			demand_unit_total_demand = demand_unit.absolute_power_consumption_array(time_consumed, time_range, for_consumer)
			Utilities.series_add(total_demand, demand_unit_total_demand, time_consumed)
			# total_demand += demand_unit_total_demand
		
		return total_demand
	
	def restore(self):
		self._consumed = False

class DemandUnit:
	'''
	props:
		- power_consumption_array | numpy array representing sequence of consumption
		- ? some way to differentiate quantitative differences between demand units
			qualitative differencesare can be handled by different time units and
			activites
	'''

	def __init__(self, power_consumption_by_time: np.ndarray):
		self._power_consumption_by_time = power_consumption_by_time
		self._length = len(self._power_consumption_by_time)

	def __len__(self):
		return self._length

	def price_effect_by_time(self, energy_price_by_time, time_range: ArrayRange, for_consumer: ActivityConsumer, for_return:np.ndarray = None) -> pd.Series:

		consumer_price_factor = for_consumer._demand_unit_price_factor[self]
		consumer_quantity_factor = for_consumer._demand_unit_quantity_factor[self]
		
		if for_return is None and len(time_range) != 1:
			for_return = np.zeros(len(time_range), dtype = np.float64)

		for start_time_step_index_delta in range(len(time_range)):
			anchour_time_step_index = time_range.start_index() + start_time_step_index_delta
			total = 0

			for time_step_index_delta, power_consumption in enumerate(self._power_consumption_by_time):
				time_step_index = anchour_time_step_index + time_step_index_delta
				if time_step_index in time_range:
					power_consumed = power_consumption / consumer_quantity_factor[time_step_index]
					effect = power_consumed * energy_price_by_time[time_step_index] * consumer_price_factor[time_step_index]
					total = total + effect
			
			if len(time_range) == 1:
				return total
			for_return[start_time_step_index_delta] += total
		
		return for_return
	
	def absolute_power_consumption_array(self, start_time_step_index, time_range: ArrayRange, for_consumer: ActivityConsumer):
		consumer_quantity_factor = for_consumer._demand_unit_quantity_factor[self]
		power_consumed_by_time = []
		for time_step_index_delta, power_consumption in enumerate(self._power_consumption_by_time):
			time_step_index = start_time_step_index + time_step_index_delta
			if time_step_index in time_range:
				quantity_factor = consumer_quantity_factor[time_step_index] if time_step_index < len(consumer_quantity_factor) else 1
				power_consumed = power_consumption / quantity_factor
				power_consumed_by_time.append(power_consumed)

		return np.array(power_consumed_by_time)

def remove_key_return(original, key, default = None):
	shallow_copy = dict(original)
	val = shallow_copy.pop(key, default)
	return shallow_copy, val

### ENVIRONMENT GENERATOR
class JsonActivityEnvironmentGenerator:	

	def generate_environment(json_file_name):
		with open(json_file_name, "r") as json_file:

			json_data = json.load(json_file)
			
			# initialize time range
			time_range_descriptor = json_data["time_range"]
			length = time_range_descriptor["length"]
			start_index = time_range_descriptor["start_index"]
			time_range = ArrayRange(start_index = start_index, length = length)

			# initialize named demand units
			named_demand_units = {}

			named_demand_units_data = json_data["named_demand_units"]
			for demand_unit_name, demand_unit_data in named_demand_units_data.items():
				demand_unit_data_array = np.array(demand_unit_data)
				new_demand_unit = DemandUnit(demand_unit_data_array)
				named_demand_units[demand_unit_name] = new_demand_unit

			# initialize activities
			named_activities = {}

			named_activities_data = json_data["activities"]
			for activity_name, activity_data in named_activities_data.items():
				if activity_name != "*":
					new_activity = Activity(activity_name)
					named_activities[activity_name] = new_activity

			activity_list = list(named_activities.values())

			# initialize activity consumers
			named_activity_consumers = {}

			named_activity_consumers_data = json_data["activity_consumers"]
			for activity_consumer_name, activity_consumer_data in named_activity_consumers_data.items():
				if activity_consumer_name != "*":
					new_activity_consumer = ActivityConsumer(activity_consumer_name)
					named_activity_consumers[activity_consumer_name] = new_activity_consumer
			
			activity_consumer_list = list(named_activity_consumers.values())

			# finalize setup of activities
			def activity_property_initilization(activity_data, activity, old_activity_data = None):
				# define demand units
				activity_demand_units = []

				activity_demand_units_data = activity_data["demand_units"]
				for elem in activity_demand_units_data:
					if isinstance(elem, list):
						demand_unit_data_array = np.array(elem)
						demand_unit = DemandUnit(demand_unit_data_array)
						# Adding to named demand units | may want to remove this feature but would reqcuire removing unnamed demand units
						named_demand_units[demand_unit] = demand_unit
					elif isinstance(elem, str):
						demand_unit = named_demand_units[elem]

					activity_demand_units.append(demand_unit)

				# define effect vectors
				activity_effect_vectors = {}

				### create actvity vectors once we know the activity consumers
				activity_effect_vectors_data = activity_data["effect_vectors"]

				### setup functions to run through json data
				generalize_effect_vector_over_times_function = JsonActivityEnvironmentGenerator.array_over_value_function(
																			time_range
																		)

				generalize_effect_vector_over_activities_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																			named_activities,
																			generalize_effect_vector_over_times_function
																		)

				generalize_effect_vector_over_consumers_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																			named_activity_consumers, 
																			generalize_effect_vector_over_activities_function
																		)

				activity_effect_vectors = generalize_effect_vector_over_consumers_function(activity_effect_vectors_data, old_dict_value = activity._effect_vectors)

				# setup activity with found information
				activity.setup(activity_demand_units, activity_effect_vectors)

			JsonActivityEnvironmentGenerator.loop_over_value(named_activities_data, named_activities, activity_property_initilization)

			# finalize setup of activity consumers			
			def activity_consumer_property_initilization(activity_consumer_data, activity_consumer, old_activity_consumer_data = None):
				
				activity_consumer_setup_args = {}

				# define activity values
				if "activity_values" in activity_consumer_data:
					activity_values_data = activity_consumer_data["activity_values"]

					#### setup functions to run through json data
					generalize_consumer_value_over_times_function = JsonActivityEnvironmentGenerator.array_over_value_function(
																				time_range
																			)

					generalize_consumer_value_over_activities_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																				named_activities,
																				generalize_consumer_value_over_times_function
																			)
					
					### create property using defined functions
				
					activity_consumer_setup_args["activity_values"] = generalize_consumer_value_over_activities_function(activity_values_data, old_dict_value = activity_consumer._activity_values)

				# define activity thresholds
				if "activity_thresholds" in activity_consumer_data:
					activity_thresholds_data = activity_consumer_data["activity_thresholds"]

					### setup functions to run through json data | First function unnecessary at the moment
					generalize_consumer_value_over_times_function = JsonActivityEnvironmentGenerator.array_over_value_function(
																				time_range
																			)

					generalize_consumer_value_over_activities_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																				named_activities,
																				generalize_consumer_value_over_times_function
																			)
					### create property using defined functions
					activity_consumer_setup_args["activity_thresholds"] = generalize_consumer_value_over_activities_function(activity_thresholds_data, old_dict_value = activity_consumer._activity_thresholds)

				# define demand unit price factors
				if "demand_unit_price_factors" in activity_consumer_data:
					demand_unit_price_factors_data = activity_consumer_data["demand_unit_price_factors"]

					### setup functions to run through json data | First function unnecessary at the moment
					generalize_consumer_value_over_times_function = JsonActivityEnvironmentGenerator.array_over_value_function(
																				time_range
																			)

					generalize_consumer_value_over_demand_units_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																				named_demand_units,
																				generalize_consumer_value_over_times_function
																			)
					### create property using defined functions
					activity_consumer_setup_args["demand_unit_price_factor"] = generalize_consumer_value_over_demand_units_function(demand_unit_price_factors_data, old_dict_value = activity_consumer._demand_unit_price_factor)

				# define demand unit quantity factors
				if "demand_unit_quantity_factors" in activity_consumer_data:
					demand_unit_quantity_factors_data = activity_consumer_data["demand_unit_quantity_factors"]

					### setup functions to run through json data | First function unnecessary at the moment
					generalize_consumer_value_over_times_function = JsonActivityEnvironmentGenerator.array_over_value_function(
																				time_range
																			)

					generalize_consumer_value_over_demand_units_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																				named_demand_units,
																				generalize_consumer_value_over_times_function
																			)

					### create property using defined functions
					activity_consumer_setup_args["demand_unit_quantity_factor"] = generalize_consumer_value_over_demand_units_function(demand_unit_quantity_factors_data, old_dict_value = activity_consumer._demand_unit_quantity_factor)

				# setup activity consumer with found information
				activity_consumer.setup(**activity_consumer_setup_args)
			
			JsonActivityEnvironmentGenerator.loop_over_value(named_activity_consumers_data, named_activity_consumers, activity_consumer_property_initilization)

			return ActivityEnvironment(activity_list, activity_consumer_list, time_range)

	# loops over values in json data and calls function on them | Effectively a map function
	def loop_over_value(property_json_data, property_name_dict, function_on_child_value = None):

		property_dict = JsonActivityEnvironmentGenerator.over_full_dict_property(property_json_data, property_name_dict, function_on_child_value)
		return list(property_dict.keys())
		

	# activites, demand_units, activity_consumers
	def dict_over_value_function(property_name_dict, function_on_child_value = None):
		def dict_over_value(property_json_data, parent = None, old_dict_value = None):

			property_dict = JsonActivityEnvironmentGenerator.over_full_dict_property(property_json_data, property_name_dict, function_on_child_value, old_dict_value)
			return property_dict

		return dict_over_value

	# time
	def array_over_value_function(array_range: ArrayRange, function_on_child_value = None):
		def array_over_value(property_json_data, parent = None, old_array_value = None):

			property_array = JsonActivityEnvironmentGenerator.over_full_array_property(property_json_data, array_range, function_on_child_value, old_array_value)
			return property_array

		return array_over_value

	# activites, demand_units, activity_consumers
	def over_full_dict_property(object_json_data, named_objects, function_on_value = None, to_return = {}):
		object_list = list(named_objects.values())

		if to_return is None:
			to_return = {} 

		object_json_data, general_object_data = remove_key_return(object_json_data, "*")

		if general_object_data is not None:
			JsonActivityEnvironmentGenerator.generalize_dict(
				general_object_data, 
				object_list, 
				to_return, 
				function_on_value
			)

		for specific_object_name, specific_object_data in object_json_data.items():
			if specific_object_name in named_objects:
				specific_object = named_objects[specific_object_name]

				if function_on_value is not None:
					to_return[specific_object] = function_on_value(specific_object_data, specific_object, to_return.get(specific_object))
				else:
					to_return[specific_object] = specific_object_data
		
		return to_return
	
	# time
	def over_full_array_property(object_json_data, array_range: ArrayRange, function_on_value = None, to_return = None):

		array_values = []
		
		object_json_data, general_object_data = remove_key_return(object_json_data, "*")

		if general_object_data is not None:

			array_values = JsonActivityEnvironmentGenerator.generalize_list(
				general_object_data, 
				array_range, 
				function_on_value
			)
			
			to_return = np.array(array_values)
		elif to_return is None:

			array_values = JsonActivityEnvironmentGenerator.generalize_list(
				None, 
				array_range
			)
			
			to_return = np.array(array_values)

		for series_key, specific_object_data in object_json_data.items():
			int_key = int(series_key)
			int_index = int_key - array_range.start_index()
			if int_index in array_range:
				if function_on_value is not None:
					to_return[int_key] = function_on_value(specific_object_data, series_key, to_return.get(int_key))
				else:
					to_return[int_key] = specific_object_data
		
		return to_return

	def generalize_dict(value_to_generalize, generalize_over, to_return = {}, function_on_value = None):
		for key in generalize_over:
			if function_on_value is not None:
				to_return[key] = function_on_value(value_to_generalize, key, to_return.get(key))
			else:
				to_return[key] = value_to_generalize
		return to_return

	def generalize_list(value_to_generalize, array_range: ArrayRange, function_on_value = None):
		series_values = []
		for index in range(array_range.start_index(), array_range.end_index()):
			if function_on_value is not None:
				series_values.append(function_on_value(value_to_generalize, index))
			else:
				series_values.append(value_to_generalize)
		return series_values

class JSONFileAutomator:
	def edit_file(json_file_name = None, reset_param = False):

		if json_file_name is None:
			json_file_name = "gym-socialgame/gym_socialgame/envs/activity_environments/activity_env.json"

		with open(json_file_name, "a+") as json_file:
			#return to start of file
			json_file.seek(0)

			if (os.stat(json_file_name).st_size != 0):
				activity_env_data = json.load(json_file)
			else:
				with open("gym-socialgame/gym_socialgame/envs/activity_environments/base_env.json", "r") as base_json_file:
					activity_env_data = json.load(base_json_file)
			
		demand_unit_dict = JSONFileAutomator.generate_demand_units_data(20, demand_units_data = activity_env_data.get("named_demand_units", {}), reset = reset_param)

		old_activities_data = activity_env_data.get("activities", {})
		old_activities_names = list(old_activities_data.keys())
		new_activities_names = JSONFileAutomator.generate_activities_names(20, activities_names = old_activities_names, reset = reset_param)
		all_activities_names = [*old_activities_names, *new_activities_names]

		old_activity_consumers_data = activity_env_data.get("activity_consumers", {})
		old_activity_consumers_names = list(old_activity_consumers_data.keys())
		new_activity_consumers_names = JSONFileAutomator.generate_activity_consumers_names(8, activity_consumers_names = old_activity_consumers_names, reset = reset_param)
		all_activity_consumers_names = [*old_activity_consumers_names, *new_activity_consumers_names]

		activity_env_data["named_demand_units"] = demand_unit_dict
		activity_env_data["activities"] = JSONFileAutomator.generate_activities_data(new_activities_names, list(demand_unit_dict.keys()), all_activity_consumers_names, old_activities_data, reset = reset_param)
		activity_env_data["activity_consumers"] = JSONFileAutomator.generate_activity_consumers_data(new_activity_consumers_names, list(demand_unit_dict.keys()), all_activities_names, old_activity_consumers_data, reset = reset_param)

		with open(json_file_name, "w") as json_file:
			json.dump(activity_env_data, json_file, ensure_ascii=False, indent=4)
	
	def generate_demand_units_data(num, demand_units_data = {}, min_length = 1, max_length = 5, min_demand = 0, max_demand = 200, reset = False):
		if reset:
			demand_units_data = {}
		for i in range(0, max(0, num - len(demand_units_data))):
			demand_unit_length = random.randint(min_length, max_length)
			letters = string.octdigits
			demand_unit_name = ''.join(random.choice(letters) for i in range(3))
			demand_unit_data = [float(random.randint(min_demand * 100, max_demand * 100)) / 100 for i in range(demand_unit_length)]
			demand_units_data[demand_unit_name] = demand_unit_data
		
		return demand_units_data

	def generate_activities_names(num, activities_names = [], reset = False):
		if reset:
			activities_names = []
		new_activities_names = []
		for i in range(0, max(0, num - len(activities_names))):
			letters = string.ascii_lowercase
			activity_name = ''.join(random.choice(letters) for i in range(3))
			new_activities_names.append(activity_name)

		return new_activities_names

	def generate_activities_data(new_activities_names, demand_units_names, activity_consumer_names, activities_data = {}, 
								min_demand_units = 1, max_demand_units = 4,
								min_activities = 0, max_activities = None,
								min_activity_consumers = 0, max_activity_consumers = None,
								min_effect = 0, max_effect = 2, mode_effect = 1, longest_effect_time_length = 10, reset = False):
		if reset:
			if "*" in activities_data:
				activities_data = {"*": activities_data["*"]}
			else:
				activities_data = {}
		all_activities_names = [*list(activities_data.keys()), *new_activities_names]
		for new_activity_name in new_activities_names:
			new_activity_data = {}

			num_demand_units = random.randint(min_demand_units, min(max_demand_units, len(demand_units_names)))
			activity_demand_unit_names = random.sample(demand_units_names, num_demand_units)
			new_activity_data["demand_units"] = activity_demand_unit_names

			effect_vector_data = {}

			effect_consumers_num = random.randint(min_activity_consumers, Utilities.special_min(max_activity_consumers, len(activity_consumer_names)))
			effect_consumer_names = random.sample(activity_consumer_names, effect_consumers_num)
			for consumer_name in effect_consumer_names:
				activity_effects = {}

				effect_activities_num = random.randint(min_activities, Utilities.special_min(max_activities, len(all_activities_names)))
				effect_activities_names = random.sample(all_activities_names, effect_activities_num)
				for activity_name in effect_activities_names:
					effect_times_num = random.randint(0, longest_effect_time_length - 1)
					effect_times_descriptors = sorted(random.sample(range(0, longest_effect_time_length), effect_times_num))
					effect_times_values = np.random.lognormal(1, 0.25, size = effect_times_num).round(decimals = 3)

					activity_effects[activity_name] = pd.Series(effect_times_values, index = effect_times_descriptors).to_dict()

				effect_vector_data[consumer_name] = activity_effects

			new_activity_data["effect_vectors"] = effect_vector_data

			activities_data[new_activity_name] = new_activity_data

		return activities_data

	def generate_activity_consumers_names(num, activity_consumers_names = [], reset = False):
		if reset:
			activity_consumers_names = []
		new_activity_consumers_names = []
		for i in range(0, max(0, num - len(activity_consumers_names))):
			letters = string.ascii_uppercase
			activity_consumer_name = ''.join(random.choice(letters) for i in range(3))
			new_activity_consumers_names.append(activity_consumer_name)

		return new_activity_consumers_names

	def generate_activity_consumers_data(new_consumers_names, demand_units_names, activities_names, activity_consumers_data = {},
										min_activities = 0, max_activities = None,
										min_demand_units = 0, max_demand_units = None,
								 		min_activity_value = 0, max_activity_value = 4, mode_activity_value = 1,
								 		min_activity_threshold = 0, max_activity_threshold = 10, mode_activity_threshold = 3,
								 		min_demand_unit_price_factor = 0, max_demand_unit_price_factor = 0.01, mode_demand_unit_price_factor = 0.002,
								 		min_demand_unit_quantity_factor = 0, max_demand_unit_quantity_factor = 2, mode_demand_unit_quantity_factor = 1,
										longest_effect_time_length = 10, reset = False):
		if reset:
			if "*" in activity_consumers_data:
				activity_consumers_data = {"*": activity_consumers_data["*"]}
			else:
				activity_consumers_data = {}
		all_activity_consumers_names = [*list(activity_consumers_data.keys()), *new_consumers_names]
		for new_consumer_name in new_consumers_names:
			new_consumer_data = {}

			# set activity values
			activity_values_data = {}
			effect_activities_values_num = random.randint(min_activities, Utilities.special_min(len(activities_names), max_activities))
			effect_activities_values_names = random.sample(activities_names, effect_activities_values_num)
			for activity_name in effect_activities_values_names:
				effect_times_num = random.randint(0, longest_effect_time_length - 1)
				effect_times_descriptors = sorted(random.sample(range(0, longest_effect_time_length), effect_times_num))
				effect_times_values = np.random.triangular(min_activity_value, mode_activity_value, max_activity_value, size = effect_times_num).round(decimals = 3)

				activity_values_data[activity_name] = pd.Series(effect_times_values, index = effect_times_descriptors).to_dict()

			new_consumer_data["activity_values"] = activity_values_data

			# set activity thresholds
			activity_thresholds_data = {}
			effect_activities_thresholds_num = random.randint(min_activities, Utilities.special_min(len(activities_names), max_activities))
			effect_activities_thresholds_names = random.sample(activities_names, effect_activities_thresholds_num)
			for activity_name in effect_activities_thresholds_names:
				effect_times_num = random.randint(0, longest_effect_time_length - 1)
				effect_times_descriptors = sorted(random.sample(range(0, longest_effect_time_length), effect_times_num))
				effect_times_values = np.random.triangular(min_activity_threshold, mode_activity_threshold, max_activity_threshold, size = effect_times_num).round(decimals = 3)

				activity_thresholds_data[activity_name] = pd.Series(effect_times_values, index = effect_times_descriptors).to_dict()

			new_consumer_data["activity_thresholds"] = activity_thresholds_data

			# set demand unit price factor
			demand_unit_price_factor_data = {}
			effect_demand_unit_price_factor_num = random.randint(min_demand_units, Utilities.special_min(len(demand_units_names), max_demand_units))
			effect_demand_unit_price_factor_names = random.sample(demand_units_names, effect_demand_unit_price_factor_num)
			for demand_unit_name in effect_demand_unit_price_factor_names:
				effect_times_num = random.randint(0, longest_effect_time_length - 1)
				effect_times_descriptors = sorted(random.sample(range(0, longest_effect_time_length), effect_times_num))
				effect_times_values = np.random.triangular(min_demand_unit_price_factor, mode_demand_unit_price_factor, max_demand_unit_price_factor, size = effect_times_num).round(decimals = 3)

				demand_unit_price_factor_data[demand_unit_name] = pd.Series(effect_times_values, index = effect_times_descriptors).to_dict()

			new_consumer_data["demand_unit_price_factor"] = demand_unit_price_factor_data

			# set demand unit quantity factor
			demand_unit_quantity_factor_data = {}
			effect_demand_unit_quantity_factor_num = random.randint(min_demand_units, Utilities.special_min(len(demand_units_names), max_demand_units))
			effect_demand_unit_quantity_factor_names = random.sample(demand_units_names, effect_demand_unit_quantity_factor_num)
			for demand_unit_name in effect_demand_unit_quantity_factor_names:
				effect_times_num = random.randint(0, longest_effect_time_length - 1)
				effect_times_descriptors = sorted(random.sample(range(0, longest_effect_time_length), effect_times_num))
				effect_times_values = np.random.triangular(min_demand_unit_quantity_factor, mode_demand_unit_quantity_factor, max_demand_unit_quantity_factor, size = effect_times_num).round(decimals = 3)

				demand_unit_quantity_factor_data[demand_unit_name] = pd.Series(effect_times_values, index = effect_times_descriptors).to_dict()

			new_consumer_data["demand_unit_price_factor"] = demand_unit_quantity_factor_data


			activity_consumers_data[new_consumer_name] = new_consumer_data

		return activity_consumers_data

if __name__ == '__main__':
	parser = argparse.ArgumentParser("reset_args")
	parser.add_argument("-f", "--file", help="The file name.")
	parser.add_argument("-r", "--reset", help="Whether file should be reset.", action="store_true")
	args = parser.parse_args()
	file_param = args.file
	reset_param = args.reset
	JSONFileAutomator.edit_file(json_file_name = file_param, reset_param = reset_param)
		