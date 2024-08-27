(:predicates
    (holding ?obj - holdable) ; whether the gripper is holding an object. If true, `free gripper` should be false.
    (open ?container - opennable)
    (free ?gripper - gripper) ; whether the gripper is not holding anything. If true, `holding` should be false.
    (on ?obj - holdable ?support - support)
    (in ?obj - holdable ?container - container)
    (under ?obj1 - object ?obj2 - object)
)