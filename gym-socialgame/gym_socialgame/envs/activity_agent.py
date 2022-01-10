import pandas as pd
import numpy as np
import json

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
  
	def execute(self, energy_price_by_time: pd.Series):
		for time_step in energy_price_by_time.index:
			for activity_consumer in self._activity_consumers:
				activity_consumer.execute_step(energy_price_by_time, time_step)
	
	def aggregate_demand(self, for_times: np.ndarray = None):
		if for_times is None:
			for_times = self._time_domain
		demand_by_activity_consumer = {}
		for activity_consumer in self._activity_consumers:
			consumer_demand = activity_consumer.aggregate_demand(for_times)
			demand_by_activity_consumer[activity_consumer] = consumer_demand
		return demand_by_activity_consumer

	def build(source_file_name = "gym-socialgame/gym_socialgame/envs/activity_env.json"):
		return JsonActivityEnvironmentGenerator.generate_environment(source_file_name)
	
	def restore(self):
		for activity in self._activities:
			activity.restore()
		for activity_consumer in self._activity_consumers:
			activity_consumer.restore()

	def execute_aggregate(self, energy_prices, for_times: np.ndarray = None):
		if for_times is None:
			for_times = self._time_domain
		energy_prices_by_time = pd.Series(energy_prices, index = for_times)
		self.execute(energy_prices_by_time)
		result = self.aggregate_demand(for_times)
		return result

	def restore_execute_aggregate(self, energy_prices, for_times: np.ndarray = None):
		if for_times is None:
			for_times = self._time_domain
		self.restore()
		result = self.execute_aggregate(energy_prices, for_times)
		return result

	def build_execute_aggregate(energy_prices, source_file_name = "gym-socialgame/gym_socialgame/envs/activity_env.json"):
		new_env : ActivityEnvironment = ActivityEnvironment.build(source_file_name)
		times = new_env._time_domain
		result = new_env.aggregate_execute(energy_prices, times)
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
		to_calculate_for_times = np.array([time_step])
		for activity, active_value_by_time in self._activity_values.items():
			if (not activity._consumed or activity._consumed is None):
				price_effect_at_time = activity.price_effect_by_time(energy_price_by_time, to_calculate_for_times, self)
				# normalized_price_effect = (price_effect[time_step] - price_effect.median())
				total_value_for_time = active_value_by_time[time_step] + price_effect_at_time
				threshold_for_time = self._activity_thresholds[activity][time_step]

				if total_value_for_time > threshold_for_time:
					self.consume(time_step, activity)

	def consume(self, time_step, activity):
		self._consumed_activities[activity] = time_step
		activity.consume(time_step)

	def aggregate_demand(self, for_times: np.ndarray):
		total_demand = pd.Series(np.full(len(for_times), 0, dtype = np.float64), index=for_times)
		for activity, time_consumed in self._consumed_activities.items():
			actvity_demand = activity.aggregate_demand(time_consumed, for_times, self)
			total_demand = total_demand.add(actvity_demand, fill_value=0)
		return total_demand

	def restore(self):
		self._consumed_activities = {}
		self._activity_values = ActivityConsumer.copy_activity_values(self._initial_activity_values)

	def copy_activity_values(activity_values):
		new_dict = {}
		for activity, time_values in activity_values.items():
			new_dict[activity] = time_values.copy(deep = True)
		return new_dict


class Activity:
	'''
	props:
		- demand_units
		- effect_vectors | a dict of ActivityConsumers to a dict of Activities to a series of effect values/functions affecting activity values by relative time
							? can also condition on the time consumed as effect may differ
		- consumed
	'''
	
	def __init__(self, demand_units = None, effect_vectors = None):
		if demand_units is None:
			if not hasattr(self, "_demand_units"):
				self._demand_units = []
		else:
			self._demand_units = demand_units

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
			
		if effect_vectors is None:
			if not hasattr(self, "_effect_vectors"):
				self._effect_vectors = {}
		else:
			self._effect_vectors = effect_vectors

	def price_effect_by_time(self, energy_price_by_time, for_times: np.ndarray, for_consumer: ActivityConsumer) -> pd.Series:
		if len(for_times) == 1:
			total = 0
			for demand_unit in self._demand_units:
				# adds price effect by time to total_price_effect
				total += demand_unit.price_effect_by_time(energy_price_by_time, for_times, for_consumer)
			return total
		else:
			total_price_effect = pd.Series(np.full(len(for_times), 0), index=for_times)
			for demand_unit in self._demand_units:
				# adds price effect by time to total_price_effect
				demand_unit.price_effect_by_time(energy_price_by_time, for_times, for_consumer, total_price_effect)
			return total_price_effect
	
	def consume(self, time_step):
		self._consumed = time_step
		for for_consumer, effect_vectors_by_activity in self._effect_vectors.items():
			for activity, effect_vector in effect_vectors_by_activity.items():
				# generate effect vector
				local_effect_vector = effect_vector.copy(deep=True)
				local_effect_vector.index = local_effect_vector.index + time_step # Add time delta to start time, note: all need to be timestamps and time deltas or all floats
				# change active values of activity by time for activity consumer
				active_values_by_time = for_consumer._activity_values[activity]
				new_active_values_by_time = active_values_by_time * local_effect_vector # change for increased complexity
				for_consumer._activity_values[activity] = new_active_values_by_time

	def aggregate_demand(self, time_consumed, for_times: np.ndarray, for_consumer: ActivityConsumer):
		total_demand = pd.Series(np.full(len(for_times), 0, dtype = np.float64), index=for_times)
		for demand_unit in self._demand_units:
			demand_unit_total_demand = demand_unit.absolute_power_consumption_array(time_consumed, for_consumer)
			total_demand = total_demand.add(demand_unit_total_demand, fill_value=0)
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

	def __init__(self, power_consumption_by_time: pd.Series):
		self._power_consumption_by_time = power_consumption_by_time

	def price_effect_by_time(self, energy_price_by_time, for_times: np.ndarray, for_consumer: ActivityConsumer, for_return:pd.Series = None) -> pd.Series:

		consumer_price_factor = for_consumer._demand_unit_price_factor[self]
		consumer_quantity_factor = for_consumer._demand_unit_quantity_factor[self]
		
		if for_return is None and len(for_times) != 1:
			for_return = pd.Series(np.full(len(for_times), 0), index=for_times)

		time_start = self._power_consumption_by_time.index[0]
		for start_time_step in for_times:
			total = 0

			for time_step_absolute, power_consumption in self._power_consumption_by_time.items():
				time_step_delta = time_step_absolute - time_start
				time_step = start_time_step + time_step_delta
				if time_step in for_times:
					power_consumed = power_consumption / consumer_quantity_factor[time_step]
					effect = power_consumed * energy_price_by_time[time_step] * consumer_price_factor[time_step]
					total = total + effect
			
			if len(for_times) == 1:
				return total
			for_return[start_time_step] += total
		
		return for_return
	
	def absolute_power_consumption_array(self, start_time_step, for_consumer: ActivityConsumer):
		consumer_quantity_factor = for_consumer._demand_unit_quantity_factor[self]
		power_consumed_by_time = []
		
		time_start = self._power_consumption_by_time.index[0]
		for time_step_absolute, power_consumption in self._power_consumption_by_time.items():
			time_step_delta = time_step_absolute - time_start
			time_step = start_time_step + time_step_delta
			power_consumed = power_consumption / consumer_quantity_factor[time_step]
			power_consumed_by_time.append(power_consumed)

		absolute_times = self._power_consumption_by_time.index + start_time_step

		return pd.Series(power_consumed_by_time, index=absolute_times)

def remove_key_return(original, key, default = None):
	shallow_copy = dict(original)
	val = shallow_copy.pop(key, default)
	return shallow_copy, val

### ENVIRONMENT GENERATOR
class JsonActivityEnvironmentGenerator:	

	def generate_environment(json_file_name):
		with open(json_file_name, "r") as json_file:

			json_data = json.load(json_file)
			
			# initialize times
			time_range_descriptor = json_data["times"]
			start = time_range_descriptor["start"]
			stop = time_range_descriptor["stop"]
			interval = time_range_descriptor["interval"]
			times = np.arange(start, stop, interval)

			# initialize named demand units
			named_demand_units = {}

			named_demand_units_data = json_data["named_demand_units"]
			for demand_unit_name, demand_unit_data in named_demand_units_data.items():
				demand_unit_data_series = pd.Series(demand_unit_data, index=times[0:len(demand_unit_data)])
				new_demand_unit = DemandUnit(demand_unit_data_series)
				named_demand_units[demand_unit_name] = new_demand_unit

			# initialize activities
			named_activities = {}

			named_activities_data = json_data["activities"]
			for activity_name, activity_data in named_activities_data.items():
				if activity_name != "*":
					new_activity = Activity()
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
						demand_unit_data_series = pd.Series(elem, index=times[0:len(elem)])
						demand_unit = DemandUnit(demand_unit_data_series)
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
				generalize_effect_vector_over_times_function = JsonActivityEnvironmentGenerator.series_over_value_function(
																			times
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
				
				actvity_consumer_setup_args = {}

				# define activity values
				if "activity_values" in activity_consumer_data:
					activity_values_data = activity_consumer_data["activity_values"]

					#### setup functions to run through json data
					generalize_consumer_value_over_times_function = JsonActivityEnvironmentGenerator.series_over_value_function(
																				times
																			)

					generalize_consumer_value_over_activities_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																				named_activities,
																				generalize_consumer_value_over_times_function
																			)
					
					### create property using defined functions
				
					actvity_consumer_setup_args["activity_values"] = generalize_consumer_value_over_activities_function(activity_values_data, old_dict_value = activity_consumer._activity_values)

				# define activity thresholds
				if "activity_thresholds" in activity_consumer_data:
					activity_thresholds_data = activity_consumer_data["activity_thresholds"]

					### setup functions to run through json data | First function unnecessary at the moment
					generalize_consumer_value_over_times_function = JsonActivityEnvironmentGenerator.series_over_value_function(
																				times
																			)

					generalize_consumer_value_over_activities_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																				named_activities,
																				generalize_consumer_value_over_times_function
																			)
					### create property using defined functions
					actvity_consumer_setup_args["activity_thresholds"] = generalize_consumer_value_over_activities_function(activity_thresholds_data, old_dict_value = activity_consumer._activity_thresholds)

				# define demand unit price factors
				if "demand_unit_price_factors" in activity_consumer_data:
					demand_unit_price_factors_data = activity_consumer_data["demand_unit_price_factors"]

					### setup functions to run through json data | First function unnecessary at the moment
					generalize_consumer_value_over_times_function = JsonActivityEnvironmentGenerator.series_over_value_function(
																				times
																			)

					generalize_consumer_value_over_demand_units_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																				named_demand_units,
																				generalize_consumer_value_over_times_function
																			)
					### create property using defined functions
					actvity_consumer_setup_args["demand_unit_price_factor"] = generalize_consumer_value_over_demand_units_function(demand_unit_price_factors_data, old_dict_value = activity_consumer._demand_unit_price_factor)

				# define demand unit quantity factors
				if "demand_unit_quantity_factors" in activity_consumer_data:
					demand_unit_quantity_factors_data = activity_consumer_data["demand_unit_quantity_factors"]

					### setup functions to run through json data | First function unnecessary at the moment
					generalize_consumer_value_over_times_function = JsonActivityEnvironmentGenerator.series_over_value_function(
																				times
																			)

					generalize_consumer_value_over_demand_units_function = JsonActivityEnvironmentGenerator.dict_over_value_function(
																				named_demand_units,
																				generalize_consumer_value_over_times_function
																			)

					### create property using defined functions
					actvity_consumer_setup_args["demand_unit_quantity_factor"] = generalize_consumer_value_over_demand_units_function(demand_unit_quantity_factors_data, old_dict_value = activity_consumer._demand_unit_quantity_factor)

				# setup activity consumer with found information
				activity_consumer.setup(**actvity_consumer_setup_args)
			
			JsonActivityEnvironmentGenerator.loop_over_value(named_activity_consumers_data, named_activity_consumers, activity_consumer_property_initilization)

			return ActivityEnvironment(activity_list, activity_consumer_list, times)

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
	def series_over_value_function(property_domain, function_on_child_value = None):
		def series_over_value(property_json_data, parent = None, old_series_value = None):

			property_series = JsonActivityEnvironmentGenerator.over_full_series_property(property_json_data, property_domain, function_on_child_value, old_series_value)
			return property_series

		return series_over_value

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
	def over_full_series_property(object_json_data, series_keys_list, function_on_value = None, to_return = None):

		series_values = []
		
		object_json_data, general_object_data = remove_key_return(object_json_data, "*")

		if general_object_data is not None:

			series_values = JsonActivityEnvironmentGenerator.generalize_list(
				general_object_data, 
				series_keys_list, 
				function_on_value
			)
			
			to_return = pd.Series(series_values, index=series_keys_list)
		elif to_return is None:

			series_values = JsonActivityEnvironmentGenerator.generalize_list(
				None, 
				series_keys_list
			)
			
			to_return = pd.Series(series_values, index=series_keys_list)

		for series_key, specific_object_data in object_json_data.items():
			int_key = int(series_key)
			if int_key in series_keys_list:
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

	def generalize_list(value_to_generalize, generalize_over, function_on_value = None):
		series_values = []
		for key in generalize_over:
			if function_on_value is not None:
				series_values.append(function_on_value(value_to_generalize, key))
			else:
				series_values.append(value_to_generalize)
		return series_values
	