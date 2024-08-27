(define (domain generic)
    (:requirements :strips :typing)
    
    ;; Defined types of objects
    (:types
        object
        robot
        holdable - object
        opennable - object
        support - object
        gripper - robot
        mug coffee-pod coffee-machine-lid - holdable
        container - opennable
        table - support
        coffee-pod-holder - container
    )
    
    ;; Defined predicates. 
    (:predicates
        (holding ?obj - holdable)
        (open ?container - opennable)
        (free ?gripper - gripper)
        (on ?obj - object ?support - support)
        (in ?obj - holdable ?container - container)
        (under ?obj1 - object ?obj2 - object)
    )

    (:action pick-up
        :parameters (?obj - holdable ?table - table ?gripper - gripper)
        :precondition (and (free ?gripper) (on ?obj ?table))
        :effect (and (holding ?obj) (not (on ?obj ?table)) (not (free ?gripper)))
    )

    (:action release-holding
        :parameters (?held - holdable ?gripper - gripper)
        :precondition (and (not (free ?gripper)) (holding ?held))
        :effect (and (free ?gripper) (not (holding ?held)))
    )
    

    (:action open-coffee-pod-holder
        :parameters (?holder - coffee-pod-holder ?lid - coffee-machine-lid ?gripper - gripper)
        :precondition (and (not (open ?holder)) (free ?gripper))
        :effect (and (open ?holder) (free ?gripper))
    )

    (:action place-pod-in-holder
        :parameters (?pod - coffee-pod ?holder - coffee-pod-holder ?gripper - gripper)
        :precondition (and (holding ?pod) (open ?holder) (not (free ?gripper)))
        :effect (and (not (holding ?pod)) (in ?pod ?holder) (free ?gripper))
    )

    (:action close-coffee-pod-holder
        :parameters (?holder - coffee-pod-holder ?lid - coffee-machine-lid ?gripper - gripper)
        :precondition (and (open ?holder) (free ?gripper))
        :effect (and (not (open ?holder)) (free ?gripper))
    )
)
