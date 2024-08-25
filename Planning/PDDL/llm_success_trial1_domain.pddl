(define (domain generic)
    (:requirements :strips :typing)
    
    ;; Define types
    (:types
        object
        robot
        holdable - object
        opennable - object
        support - object
        gripper - robot
        mug coffee-pod coffee-dispenser-lid - holdable
        container - opennable
        table - support
        drawer coffee-dispenser - container
    )
    
    ;; Define predicates
    (:predicates
        (holding ?obj - holdable) ; whether the gripper is holding an object. If true, `free gripper` should be false.
        (open ?container - opennable)
        (free ?gripper - gripper) ; whether the gripper is not holding anything. If true, `holding` should be false.
        (on ?obj - holdable ?support - support)
        (in ?obj - holdable ?container - container)
        (under ?obj1 - object ?obj2 - object)
    )
    
    ;; Define actions using the predicates given
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
    

    (:action open-dispenser
        :parameters (?dispenser - coffee-dispenser ?lid - coffee-dispenser-lid ?gripper - gripper)
        :precondition (and (not (open ?dispenser)) (free ?gripper))
        :effect (and (open ?dispenser) (free ?gripper))
    )

    (:action place-pod-in-dispenser
        :parameters (?pod - coffee-pod ?dispenser - coffee-dispenser ?gripper - gripper)
        :precondition (and (holding ?pod) (open ?dispenser) (not (free ?gripper)))
        :effect (and (not (holding ?pod)) (in ?pod ?dispenser) (free ?gripper))
    )

    (:action close-dispenser
        :parameters (?dispenser - coffee-dispenser ?lid - coffee-dispenser-lid ?gripper - gripper)
        :precondition (and (open ?dispenser) (free ?gripper))
        :effect (and (not (open ?dispenser)) (free ?gripper))
    )

    (:action remove-from-drawer
        :parameters (?pod - coffee-pod ?drawer - drawer ?gripper - gripper)
        :precondition (and (in ?pod ?drawer) (free ?gripper))
        :effect (and (holding ?pod) (not (in ?pod ?drawer)) (not (free ?gripper)))
    )

    (:action place-under-dispenser
        :parameters (?mug - mug ?dispenser - coffee-dispenser ?gripper - gripper)
        :precondition (and (holding ?mug) (not (free ?gripper)))
        :effect (and (under ?mug ?dispenser) (not (holding ?mug)) (free ?gripper))
    )
    
    
)
