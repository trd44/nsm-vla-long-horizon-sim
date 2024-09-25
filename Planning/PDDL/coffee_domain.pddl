

(define (domain coffee)
    (:requirements :strips :typing)
    (:types
        mug coffee-pod coffee-pod-holder coffee-machine-lid drawer table gripper - object
    )

    ;; Define predicates
    (:predicates
        (can-reach ?obj - object)
        (can-hold ?obj - object)
        (can-open ?obj - object)
        (can-contain ?container - object ?obj - object)
        (can-flip ?lid - object)
        (on-table ?obj - object ?table - table)
        (holding ?obj - object) ; whether the gripper is holding an object. If true, `free gripper` should be false.
        (occupying-gripper ?obj - object) ; whether the object is occupying the gripper. If true, `free gripper` should be false.
        (reached ?obj - object) ; whether the gripper is within 1 unit of the object. If true, `can-reach` should be true.
        (attached ?obj1 - object ?obj2 - object) ; whether obj1 is attached to obj2. 
        (in ?obj - object ?container - object)
        (open ?container - object)
        (free ?gripper - gripper) ; whether the gripper is not holding anything. If true, `holding` should be false.
        (under ?bottom - object ?top - object)
    )
    
    ;; Define actions using the predicates given
    (:action pick-up-tabletop
        :parameters (?obj - object ?table - table ?gripper - gripper) 
        :precondition (and (on-table ?obj ?table) (can-reach ?obj) (can-hold ?obj) (free ?gripper)) 
        :effect (and (reached ?obj) (holding ?obj) (occupying-gripper ?obj) (not (on-table ?obj ?table)) (not (free ?gripper))) 
    )
    
    (:action open-coffee-pod-holder
        :parameters (?holder - coffee-pod-holder ?lid - coffee-machine-lid ?gripper - gripper)
        :precondition (and (not (open ?holder)) (can-reach ?lid) (can-flip ?lid) (can-open ?holder) (attached ?lid ?holder) (free ?gripper))
        :effect (and (reached ?lid) (occupying-gripper ?lid) (open ?holder) (not (free ?gripper)))
    )


    (:action close-coffee-pod-holder
        :parameters (?holder - coffee-pod-holder ?lid - coffee-machine-lid ?gripper - gripper)
        :precondition (and (open ?holder) (can-reach ?lid) (can-flip ?lid) (attached ?lid ?holder) (free ?gripper))
        :effect (and (reached ?lid) (occupying-gripper ?lid) (not (open ?holder)) (not (free ?gripper)))
    )

    (:action free-gripper
        :parameters (?gripper - gripper)
        :precondition (and (not (free ?gripper)))
        :effect (and (free ?gripper))
    )
    

    (:action place-pod-in-holder
        :parameters (?pod - coffee-pod ?holder - coffee-pod-holder ?gripper - gripper)
        :precondition (and (holding ?pod) (open ?holder) (can-contain ?holder ?pod) (can-reach ?pod)(not (free ?gripper)))
        :effect (and (reached ?pod) (not (holding ?pod)) (in ?pod ?holder) (free ?gripper))
    )


    (:action place-mug-under-holder
        :parameters (?mug - mug ?holder - coffee-pod-holder ?gripper - gripper)
        :precondition (and (holding ?mug) (occupying-gripper ?mug) (not (free ?gripper)) (can-reach ?holder))
        :effect (and (under ?mug ?holder) (not (holding ?mug)) (free ?gripper))
    )

)