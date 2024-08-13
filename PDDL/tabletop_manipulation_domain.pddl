(define (domain tabletop-manipulation-domain)
    (:requirements :strips :typing)
    (:types 
        object
        holdable - object
        opennable - object
        nothing mug coffee-pod - holdable
        drawer coffee-machine - opennable
    )
    
    (:predicates
        (open ?obj - object)
        (holding ?obj - object)
        (in-drawer ?obj - object)
        (pod-installed ?pod - coffee-pod)
    )
    
    (:action pick-up
        :parameters (?holdable - holdable)
        :precondition (and (holding nothing) (not (in-drawer ?holdable)))
        :effect (and (holding ?holdable) (not (holding nothing)))
    )
    
    (:action put-in-drawer
        :parameters (?holdable - holdable ?drawer - drawer)
        :precondition (and (holding ?holdable) (open ?drawer))
        :effect (and (in-drawer ?holdable) (not (holding ?holdable)) (holding nothing))
    )
    
    (:action open
        :parameters (?opennable - opennable)
        :precondition (and (not (open ?opennable)) (holding nothing))
        :effect (open ?opennable)
    )
    
    (:action close
        :parameters (?opennable - opennable)
        :precondition (and (open ?opennable) (holding nothing))
        :effect (not (open ?opennable))
    )
    
    (:action put-in-coffee-pod
        :parameters (?pod - coffee-pod ?machine - coffee-machine)
        :precondition (and (holding ?pod) (open ?machine))
        :effect (and (pod-installed ?pod) (not (holding ?pod)) (holding nothing))
    )
)
